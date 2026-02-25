"""
=============================================================================
Server-friendly preprocessing for CLUE/L1000 level3 gctx
Output format: HGNC-token sequence + rank-based binned expression (+ metadata)
=============================================================================

Requirements (HPC / server):
  - numpy, pandas, h5py, tqdm

=============================================================================
Pipeline overview (3 core steps)
=============================================================================

Step 0: Load metadata
  - Load conditional table (siginfo), filter qc_pass==1
  - Merge cellinfo (cell_lineage, primary_disease, donor_sex, donor_age)
  - Merge compoundinfo (pert_id -> cmap_name fallback)
  - Create standardized metadata columns matching the single-cell pipeline:
      tissue, region, sampleType, compartment, cellType1, cellType2,
      majorCluster, subCluster, sex, age, perturbType, disease, drug, dose, time

Step 1: Build gene mapping + vocabulary
  - gctx stores Entrez IDs as row identifiers
  - Mapping chain (primary):  Entrez → (geneinfo) → Ensembl → (HGNC table) → HGNC numeric ID
  - Mapping chain (fallback): Entrez → (geneinfo) → gene_symbol → (HGNC alias) → HGNC numeric ID
  - Build vocabulary (dict format, aligned with single-cell pipeline):
      {"<pad>": 0, "<cls>": 1, "5": 2, ...}  (keys are HGNC numeric ID strings)
  - Outputs:
      * gene_vocab.json                      (HGNC numeric ID → token_id)
      * conn_id_to_hgnc.json                 (HGNC numeric ID → gene symbol, for interpretation)
      * gene_map_gctxindex_to_hgnctoken.npy  (gctx gene index → token_id)
      * gctx_gene_to_hgncid.tsv             (readable debug table)
      * genes_entrez.json                    (Entrez IDs reference)
      * genes_symbol.json                    (gene symbols reference)

Step 2: Tokenize + shard
  - Stream over gctx columns (signatures) in chunks
  - For each signature:
      1) Select Top-(K+EXTRA) genes by abs(expression)
      2) Map gctx gene indices → BMG token ids
      3) Drop unmapped genes (token==0), keep first K
      4) Rank-based binning: higher bin = higher expression, bin 0 = padding
      5) Pad to K with <pad>=0
  - Outputs:
      * shards/shard_XXXXX.npz
      * domain_map.json

Each shard contains:
  gene_ids  int32   [B, K]   - HGNC token ids (0 = <pad>)
  bin_ids   int16   [B, K]   - rank-based bins (0 = padding, 1~n_bins = real)
  lengths   int16   [B]      - non-PAD count per row
  text_data U       [B, C]   - standard metadata columns (unicode, aligned with single-cell)
  text_cols U       [C]      - column names
  domain_id int32   [B]      - coarse domain id
  sample_id U       [B]      - signature id

Continuous perturbation values (dose_um, dose_raw, time_h) are computed
separately by continuous_parse.py and added to shards as a post-processing step.

=============================================================================
Design notes
=============================================================================

Gene ID alignment:
  - Vocabulary keys are HGNC numeric IDs (e.g., "5" from "HGNC:5"), identical
    to the single-cell pipeline (data_process_v0.0.py). This allows multi-modal
    pretraining where L1000 signatures and single-cell data share the same gene
    token space.

Text data alignment:
  - Uses the same 15 standard columns as the single-cell pipeline.
  - Unavailable fields (e.g. cellType1 for L1000) are filled with "none".
  - Stored as 2D unicode array [B, C] with separate text_cols header,
    NOT as concatenated "k=v | k=v" strings.
  - dose column uses pert_idose directly (e.g. "6.66 uM", "20 uL", "0.1 ng/ml"),
    so the model can infer unit information from text_data directly.
  - time column uses pert_itime directly (e.g. "24 h", "72 h").

Binning convention (matching v0.0):
  - bin = n_bins - (rank * n_bins) // L
  - Range: 1 (lowest expression) to n_bins (highest expression), 0 = padding.
  - This is consistent with the single-cell pipeline where bin 0 is reserved
    for padding positions.

=============================================================================
Examples
=============================================================================

python data_process_L1000_v0.1.py \
  --gctx level3_beta_trt_cp_n1805898x12328.gctx \
  --conditional siginfo_beta.txt \
  --cellinfo cellinfo_beta.txt \
  --compoundinfo compoundinfo_beta.txt \
  --geneinfo geneinfo_beta.txt \
  --hgnc hgnc_complete_set.txt \
  --out_dir out_L1000 \
  --use_abs
=============================================================================
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def norm_ensembl(x: str) -> str:
    """
    Normalize Ensembl gene ID:
      - Strip whitespace
      - Drop version suffix (e.g. ENSG00000001234.5 → ENSG00000001234)
      - Return "" for invalid/missing values
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "na", "none"):
        return ""
    if s.startswith("ENS") and "." in s:
        s = s.split(".", 1)[0]
    return s


def safe_str(x) -> str:
    """Convert value to string; treat NaN/None/empty as 'none'."""
    if x is None:
        return "none"
    s = str(x)
    if s == "" or s.lower() in ("nan", "none"):
        return "none"
    return s


def encode_labels(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode categorical labels as integers (LabelEncoder).

    Returns:
        y: Encoded integer array
        mapping: {label_name: integer_ID}
    """
    vals = series.astype(str).fillna("none").values
    uniq = pd.unique(vals).tolist()
    mapping = {k: i for i, k in enumerate(uniq)}
    y = np.array([mapping[v] for v in vals], dtype=np.int32)
    return y, mapping



# =============================================================================
# Gene Mapping: gctx Entrez → Ensembl → HGNC numeric ID
# =============================================================================

def step_build_gene_mapping(
    gctx_entrez: List[str],
    geneinfo_path: Path,
    hgnc_path: Path,
    out_dir: Path,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Step 1: Build mapping from gctx gene index to HGNC token ID.

    Mapping chain (primary):
        gctx gene index → Entrez ID → (geneinfo) → Ensembl ID → (HGNC table) → HGNC numeric ID

    Fallback (when Ensembl mapping fails):
        Entrez ID → (geneinfo) → gene_symbol → (HGNC symbol/alias/prev_symbol) → HGNC numeric ID

    Vocabulary format (aligned with data_process_v0.0.py):
        {"<pad>": 0, "<cls>": 1, "5": 2, "7": 3, ...}
        (keys are HGNC numeric ID strings extracted from "HGNC:5")

    Args:
        gctx_entrez: List of Entrez IDs from gctx ROW metadata
        geneinfo_path: Path to geneinfo_beta.txt (Entrez → Ensembl + gene_symbol)
        hgnc_path: Path to hgnc_complete_set.txt (tab-separated)
        out_dir: Output directory

    Returns:
        map_arr: int32 array of length G, map_arr[g] = HGNC token id (0 = <pad>/unmapped)
        vocab: {HGNC_numeric_id_str: token_id} dictionary

    Outputs:
        gene_vocab.json                      - HGNC numeric ID → token_id (dict)
        conn_id_to_hgnc.json                 - HGNC numeric ID → gene symbol (for interpretation)
        gene_map_gctxindex_to_hgnctoken.npy  - gctx gene index → token id
        gctx_gene_to_hgncid.tsv             - Readable debug table
    """
    # ---- geneinfo: Entrez → Ensembl + gene_symbol ----
    gi = pd.read_csv(geneinfo_path, sep="\t", dtype=str,
                     usecols=["gene_id", "ensembl_id", "gene_symbol"])
    gi["entrez_id"] = gi["gene_id"].astype(str)
    gi["ensembl_norm"] = gi["ensembl_id"].apply(norm_ensembl)

    # ---- HGNC: Load table ----
    hgnc_df = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    print(f"[HGNC] Loaded {len(hgnc_df):,} rows from {hgnc_path.name}")

    # Extract HGNC numeric ID: "HGNC:5" → "5"
    hgnc_df["hgnc_num"] = hgnc_df["hgnc_id"].apply(
        lambda x: str(x).strip().split(":", 1)[-1]
        if pd.notna(x) and ":" in str(x) else ""
    )

    # ---- HGNC: Ensembl → HGNC numeric ID (primary lookup) ----
    hgnc_df["ensembl_norm"] = hgnc_df["ensembl_gene_id"].apply(norm_ensembl)
    hgnc_ens = hgnc_df.loc[
        (hgnc_df["ensembl_norm"] != "") & (hgnc_df["hgnc_num"] != ""),
        ["ensembl_norm", "hgnc_num"]
    ].drop_duplicates("ensembl_norm", keep="first")

    # ---- HGNC: Symbol/alias/prev_symbol → HGNC numeric ID (fallback lookup) ----
    hgnc_valid = hgnc_df[hgnc_df["hgnc_num"] != ""].copy()
    symbol_to_hgnc_num: Dict[str, str] = {}

    sym_df = hgnc_valid[["symbol", "hgnc_num"]].dropna(subset=["symbol"]).copy()
    sym_df["symbol"] = sym_df["symbol"].str.strip()
    sym_df = sym_df[sym_df["symbol"] != ""]
    symbol_to_hgnc_num.update(dict(zip(sym_df["symbol"], sym_df["hgnc_num"])))
    symbol_to_hgnc_num.update(dict(zip(sym_df["symbol"].str.upper(), sym_df["hgnc_num"])))

    for col in ("alias_symbol", "prev_symbol"):
        if col not in hgnc_valid.columns:
            continue
        a_df = hgnc_valid[["hgnc_num", col]].dropna(subset=[col]).copy()
        a_df[col] = a_df[col].str.strip().str.strip('"')
        a_df = a_df[a_df[col] != ""]
        a_exp = a_df.assign(**{col: a_df[col].str.split("|")}).explode(col)
        a_exp[col] = a_exp[col].str.strip()
        a_exp = a_exp[a_exp[col] != ""]
        symbol_to_hgnc_num.update(dict(zip(a_exp[col], a_exp["hgnc_num"])))
        symbol_to_hgnc_num.update(dict(zip(a_exp[col].str.upper(), a_exp["hgnc_num"])))

    print(f"[GENE MAP] HGNC Ensembl lookup: {len(hgnc_ens):,} entries, "
          f"symbol lookup: {len(symbol_to_hgnc_num):,} entries")

    # ---- Join: Entrez → Ensembl → HGNC numeric ID (primary) ----
    # Keep ALL genes (including those without Ensembl ID) so symbol fallback can apply
    gi2 = gi[["entrez_id", "ensembl_norm", "gene_symbol"]].copy()
    gi2 = gi2.merge(hgnc_ens, on="ensembl_norm", how="left")
    # Rows with ensembl_norm=="" or no HGNC match → hgnc_num is NaN

    # ---- Symbol fallback: gene_symbol → HGNC symbol/alias → HGNC numeric ID ----
    if symbol_to_hgnc_num:
        primary_failed = gi2["hgnc_num"].isna() | gi2["hgnc_num"].str.lower().isin(["nan", "none", ""])
        if primary_failed.any():
            gi2["_sym"] = gi2["gene_symbol"].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip().lower() not in ("nan", "none", "") else ""
            )
            can_fallback = primary_failed & (gi2["_sym"] != "")
            gi2.loc[can_fallback, "hgnc_num"] = gi2.loc[can_fallback, "_sym"].map(symbol_to_hgnc_num)
            gi2.drop(columns=["_sym"], inplace=True)

            still_failed = gi2["hgnc_num"].isna() | gi2["hgnc_num"].str.lower().isin(["nan", "none", ""])
            n_ensembl = (~primary_failed).sum()
            n_symbol  = (primary_failed & ~still_failed).sum()
            print(f"[GENE MAP] Mapped via Ensembl: {n_ensembl:,}, via symbol fallback: {n_symbol:,}, still unmapped: {still_failed.sum():,}")

    entrez_to_hgncid = dict(zip(gi2["entrez_id"], gi2["hgnc_num"]))
    entrez_to_symbol = dict(zip(gi2["entrez_id"], gi2["gene_symbol"]))

    # ---- Build vocabulary (dict format, matching v0.0) ----
    # Collect all valid HGNC numeric IDs that appear in gctx genes
    hgnc_ids_in_gctx = []
    for e in gctx_entrez:
        hid = entrez_to_hgncid.get(str(e), None)
        if hid is not None and str(hid).lower() not in ("nan", "none", ""):
            hgnc_ids_in_gctx.append(str(hid))
    # Sort numerically for consistent ordering
    uniq_hgnc_ids = sorted(set(hgnc_ids_in_gctx), key=lambda x: int(x) if x.isdigit() else 0)

    # vocab: <pad>=0, <cls>=1, HGNC_numeric_id_str=2...
    vocab: Dict[str, int] = {"<pad>": 0, "<cls>": 1}
    for hid in uniq_hgnc_ids:
        vocab[hid] = len(vocab)

    (out_dir / "gene_vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"[VOCAB] Saved vocab (size={len(vocab)}) -> {out_dir / 'gene_vocab.json'}")

    # ---- Build conn_id_to_hgnc (HGNC numeric ID → gene symbol, for interpretation) ----
    conn_to_hgnc: Dict[str, str] = {"<pad>": "<pad>", "<cls>": "<cls>"}
    for e in gctx_entrez:
        hid = entrez_to_hgncid.get(str(e), None)
        sym = entrez_to_symbol.get(str(e), "")
        if hid and str(hid) in vocab and str(hid) not in conn_to_hgnc:
            conn_to_hgnc[str(hid)] = str(sym) if sym else ""
    (out_dir / "conn_id_to_hgnc.json").write_text(json.dumps(conn_to_hgnc, indent=2))
    print(f"[VOCAB] Saved conn_id_to_hgnc -> {out_dir / 'conn_id_to_hgnc.json'}")

    # ---- Build map array: gctx gene index → token id ----
    G = len(gctx_entrez)
    map_arr = np.zeros((G,), dtype=np.int32)  # 0 = <pad>/unmapped
    miss = 0
    for g, e in enumerate(gctx_entrez):
        hid = entrez_to_hgncid.get(str(e), None)
        if hid is None or str(hid).lower() in ("nan", "none", ""):
            miss += 1
            continue
        map_arr[g] = vocab.get(str(hid), 0)

    np.save(out_dir / "gene_map_gctxindex_to_hgnctoken.npy", map_arr)
    print(f"[GENE MAP] gctx genes={G:,}, mapped={G - miss:,}, unmapped={miss:,}")

    # ---- Debug table ----
    tmp = pd.DataFrame({
        "gctx_gene_index": np.arange(G),
        "entrez_id": [str(x) for x in gctx_entrez]
    })
    tmp = tmp.merge(
        gi2[["entrez_id", "ensembl_norm", "gene_symbol", "hgnc_num"]],
        on="entrez_id", how="left"
    )
    tmp["hgnctoken"] = map_arr
    tmp.to_csv(out_dir / "gctx_gene_to_hgncid.tsv", sep="\t", index=False)
    print(f"[GENE MAP] Saved debug table -> gctx_gene_to_hgncid.tsv")

    return map_arr, vocab


# =============================================================================
# Metadata Helpers
# =============================================================================

def coarse_domain_str(row: pd.Series) -> str:
    """
    Build coarse domain string for downstream domain-aware modeling.

    Format: bead_batch|det_plate|rna_plate|project_code
    """
    bead_batch   = safe_str(row.get("bead_batch", "none"))
    det_plate    = safe_str(row.get("det_plate", "none"))
    rna_plate    = safe_str(row.get("rna_plate", "none"))
    project_code = safe_str(row.get("project_code", "none"))
    return "|".join([bead_batch, det_plate, rna_plate, project_code])


# =============================================================================
# Core Steps
# =============================================================================

def step_load_metadata(
    conditional_path: Path,
    cellinfo_path: Path,
    compoundinfo_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Step 0: Load and merge L1000 metadata, filter qc_pass==1.

    Process:
      1. Load conditional table, filter qc_pass==1
      2. Merge cellinfo (cell_lineage, primary_disease, donor_sex, donor_age)
      3. Create standardized columns matching single-cell pipeline:
           tissue, region, sampleType, compartment, cellType1, cellType2,
           majorCluster, subCluster, sex, age, perturbType, disease, drug, dose, time
      4. Load compoundinfo for drug name fallback

    Returns:
        cond_keep: DataFrame indexed by sample_id, with standardized columns
        pertid_to_cmap: {pert_id: cmap_name} fallback dict
    """
    # ---- Load conditional ----
    cond_cols = [
        "sample_id", "qc_pass",
        "pert_id", "cmap_name", "pert_type",
        "nearest_dose", "pert_dose", "pert_dose_unit",
        "pert_idose", "pert_time", "pert_itime",
        "cell_iname",
        "bead_batch", "det_plate", "rna_plate", "project_code",
    ]
    cond = pd.read_csv(conditional_path, sep="\t", dtype=str,
                       usecols=lambda c: c in cond_cols)
    cond_keep = cond[(cond["qc_pass"] == "1") & cond["sample_id"].notna()].copy()
    print(f"[META] conditional rows: {len(cond):,}")
    print(f"[META] keep qc_pass==1:  {len(cond_keep):,}")

    # ---- Merge cellinfo ----
    cell = pd.read_csv(cellinfo_path, sep="\t", dtype=str,
                       usecols=["cell_iname", "cell_lineage", "primary_disease",
                                "donor_sex", "donor_age"])
    cond_keep = cond_keep.merge(cell, on="cell_iname", how="left")

    # ---- Standardize column names (matching v0.0 text_cols) ----
    # Available from L1000 data:
    cond_keep["tissue"]      = cond_keep["cell_lineage"].apply(safe_str)
    cond_keep["sex"]         = cond_keep["donor_sex"].apply(safe_str)
    cond_keep["age"]         = cond_keep["donor_age"].apply(safe_str)
    cond_keep["perturbType"] = cond_keep["pert_type"].apply(safe_str)
    cond_keep["disease"]     = cond_keep["primary_disease"].apply(safe_str)

    # Not available in L1000 (single-cell specific):
    cond_keep["region"]       = "none"
    cond_keep["sampleType"]   = "none"
    cond_keep["compartment"]  = "none"
    cond_keep["cellType1"]    = "none"
    cond_keep["cellType2"]    = "none"
    cond_keep["majorCluster"] = "none"
    cond_keep["subCluster"]   = "none"

    # ---- Drug: cmap_name with fallback to compoundinfo ----
    cmpd = pd.read_csv(compoundinfo_path, sep="\t", dtype=str,
                       usecols=["pert_id", "cmap_name"])
    pertid_to_cmap = dict(zip(cmpd["pert_id"].astype(str),
                              cmpd["cmap_name"].astype(str)))

    # Vectorized drug resolution
    drug_col = cond_keep["cmap_name"].apply(safe_str)
    fallback_drug = cond_keep["pert_id"].apply(safe_str).map(pertid_to_cmap).apply(safe_str)
    cond_keep["drug"] = drug_col.where(drug_col != "none", fallback_drug)

    # ---- Dose (text): use pert_idose directly (e.g. "6.66 uM", "20 uL") ----
    cond_keep["dose"] = cond_keep["pert_idose"].apply(safe_str)

    # ---- Time (text): use pert_itime directly (e.g. "24 h", "72 h") ----
    cond_keep["time"] = cond_keep["pert_itime"].apply(safe_str)

    # ---- Index by sample_id ----
    cond_keep.set_index("sample_id", inplace=True)

    return cond_keep, pertid_to_cmap


def step_tokenize_shards(
    gctx_path: Path,
    out_dir: Path,
    map_g_to_hgnctoken: np.ndarray,
    cond_keep: pd.DataFrame,
    # hyperparameters
    topk: int,
    n_bins: int,
    extra: int,
    chunk_rows: int,
    shard_size: int,
    use_abs: bool,
    # metadata
    text_cols: Optional[str] = None,
) -> Dict[str, int]:
    """
    Step 2: Tokenize all signatures and write shards.

    Tokenization (per signature):
      1. Select Top-(K+EXTRA) genes by abs(expression)
      2. Map gctx gene indices → HGNC token ids
      3. Drop unmapped genes (token==0), keep first K
      4. Rank-based binning: higher bin = higher expression, bin 0 = padding
      5. Pad to K with <pad>=0

    Binning convention (matching v0.0):
      bin = n_bins - (rank * n_bins) // L
      Range: 1 (lowest expression) to n_bins (highest), 0 = padding.

    Args:
        gctx_path: Path to gctx file
        out_dir: Output directory
        map_g_to_hgnctoken: int32[G], gctx gene index → HGNC token id (0=unmapped)
        cond_keep: Metadata DataFrame indexed by sample_id (standardized columns)
        topk: Max genes per signature = sequence length
        n_bins: Number of expression bins
        extra: Overselect TOPK+EXTRA genes to handle unmapped
        chunk_rows: Rows per gctx read chunk
        shard_size: Signatures per output shard
        use_abs: Use abs(expression) for gene ranking
        text_cols: Comma-separated metadata columns to save

    Returns:
        domain_map: {domain_string: domain_id} mapping
    """
    # ---- Parse text_cols ----
    text_cols_list = None
    if text_cols is not None and str(text_cols).strip() != "":
        text_cols_list = [c.strip() for c in text_cols.split(",") if c.strip()]

        text_cols_exist = [c for c in text_cols_list if c in cond_keep.columns]
        text_cols_missing = [c for c in text_cols_list if c not in cond_keep.columns]

        if text_cols_missing:
            print(f"[TOK] Warning: missing columns will be filled with 'none': {text_cols_missing}")
        print(f"[TOK] Will save text_data columns: {text_cols_list}")
        print(f"[TOK]   - existing: {text_cols_exist}")
        print(f"[TOK]   - missing (fill 'none'): {text_cols_missing}")

    # ---- Buffers ----
    global_domain_to_id: Dict[str, int] = {}
    shard_id = 0
    buf_gene: List[np.ndarray] = []
    buf_bin: List[np.ndarray] = []
    buf_len: List[np.ndarray] = []
    buf_text: List[np.ndarray] = []      # list of [B_i, C] arrays
    buf_domain: List[int] = []
    buf_sid: List[str] = []
    buf_count = 0

    shards_dir = out_dir / "shards"
    ensure_dir(shards_dir)

    TOPK = topk
    EXTRA = extra
    N_BINS = n_bins

    with h5py.File(str(gctx_path), "r") as f:
        X = f["/0/DATA/0/matrix"]
        cid_ds = f["/0/META/COL/id"]

        N, G = X.shape
        print(f"[GCTX] matrix shape: (N={N:,}, G={G:,})")

        K0 = min(G, TOPK + EXTRA)

        pbar = tqdm(range(0, N, chunk_rows), desc="[TOK] chunking gctx")
        for start in pbar:
            end = min(start + chunk_rows, N)

            # ---- Filter: only keep qc_pass==1 signatures ----
            cids = [x.decode("utf-8") for x in cid_ds[start:end]]
            idx_pd = pd.Index(cids)
            in_mask = idx_pd.isin(cond_keep.index)
            if not in_mask.any():
                continue

            keep_pos = np.where(in_mask.to_numpy())[0]
            keep_ids = [cids[i] for i in keep_pos]

            meta = cond_keep.loc[keep_ids].copy()
            meta = meta.reindex(keep_ids)

            # ---- Read expression ----
            block = X[start:end, :][keep_pos, :]   # (B, G)
            score = np.abs(block) if use_abs else block

            # ---- Top-(K+EXTRA) selection (vectorized) ----
            idx_top = np.argpartition(-score, K0 - 1, axis=1)[:, :K0]
            top_vals = np.take_along_axis(score, idx_top, axis=1)
            order = np.argsort(-top_vals, axis=1)
            idx_sorted = np.take_along_axis(idx_top, order, axis=1).astype(np.int32)

            # ---- Map to HGNC tokens ----
            bmg_tokens_all = map_g_to_hgnctoken[idx_sorted]  # (B, K0), 0=unmapped

            # ---- Keep first TOPK mapped genes; pad with 0 if needed ----
            B = bmg_tokens_all.shape[0]
            out_gene = np.zeros((B, TOPK), dtype=np.int32)
            out_bin  = np.zeros((B, TOPK), dtype=np.int16)
            out_len  = np.zeros((B,), dtype=np.int16)

            for i in range(B):
                tok = bmg_tokens_all[i]
                tok = tok[tok != 0]             # drop unmapped
                L = min(TOPK, tok.shape[0])
                if L > 0:
                    out_gene[i, :L] = tok[:L]
                    # Rank-based bins (matching v0.0 convention):
                    # rank 0 (highest expression) → bin N_BINS
                    # rank L-1 (lowest expression) → bin 1
                    # padding positions remain 0
                    ranks = np.arange(L, dtype=np.int32)
                    bins = (N_BINS - (ranks * N_BINS) // L).astype(np.int16)
                    out_bin[i, :L] = bins
                out_len[i] = L

            buf_gene.append(out_gene)
            buf_bin.append(out_bin)
            buf_len.append(out_len)

            # ---- Build text_data (2D unicode array, matching v0.0 format) ----
            if text_cols_list is not None:
                meta_dict = {}
                for col in text_cols_list:
                    if col in meta.columns:
                        meta_dict[col] = meta[col].apply(safe_str).values
                    else:
                        meta_dict[col] = np.full(B, "none", dtype=object)
                text_chunk = pd.DataFrame(meta_dict).astype("string").fillna("none")
                text_arr = np.asarray(text_chunk.to_numpy(dtype=str), dtype="U")
                buf_text.append(text_arr)

            # ---- Domain IDs + sample IDs ----
            for i in range(B):
                row = meta.iloc[i]
                dom_str = coarse_domain_str(row)
                if dom_str not in global_domain_to_id:
                    global_domain_to_id[dom_str] = len(global_domain_to_id)
                buf_domain.append(global_domain_to_id[dom_str])
                buf_sid.append(keep_ids[i])

            buf_count += B

            # ---- Write shard if buffer is full ----
            if buf_count >= shard_size:
                _write_shard(
                    shards_dir / f"shard_{shard_id:05d}.npz",
                    buf_gene, buf_bin, buf_len, buf_text,
                    buf_domain, buf_sid, text_cols_list,
                )
                shard_id += 1
                buf_gene, buf_bin, buf_len = [], [], []
                buf_text = []
                buf_domain, buf_sid = [], []
                buf_count = 0

        # ---- Write remaining ----
        if buf_count > 0:
            _write_shard(
                shards_dir / f"shard_{shard_id:05d}.npz",
                buf_gene, buf_bin, buf_len, buf_text,
                buf_domain, buf_sid, text_cols_list,
            )

    return global_domain_to_id


def _write_shard(
    path: Path,
    buf_gene: List[np.ndarray],
    buf_bin: List[np.ndarray],
    buf_len: List[np.ndarray],
    buf_text: List[np.ndarray],
    buf_domain: List[int],
    buf_sid: List[str],
    text_cols_list: Optional[List[str]],
) -> None:
    """
    Write one compressed npz shard.

    Uses unicode dtype ("U") for string data instead of object dtype,
    matching the single-cell pipeline convention.
    """
    gene_ids = np.concatenate(buf_gene, axis=0)
    bin_ids  = np.concatenate(buf_bin, axis=0)
    lengths  = np.concatenate(buf_len, axis=0)

    payload = dict(
        gene_ids=gene_ids,
        bin_ids=bin_ids,
        lengths=lengths,
        domain_id=np.array(buf_domain, dtype=np.int32),
        sample_id=np.asarray(buf_sid, dtype="U"),
    )

    if text_cols_list is not None and buf_text:
        text_data = np.concatenate(buf_text, axis=0)
        payload["text_data"] = text_data
        payload["text_cols"] = np.asarray(text_cols_list, dtype="U")

    np.savez_compressed(str(path), **payload)
    print(f"[SHARD] {path.name}  n={gene_ids.shape[0]:,}")


# =============================================================================
# Command Line Interface (CLI)
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Argument groups:
        Input: --gctx, --conditional, --cellinfo, --compoundinfo, --geneinfo
        HGNC mapping: --hgnc
        Output: --out_dir
        Hyperparameters: --topk, --n_bins, --chunk_rows, --shard_size, --extra
        Metadata: --text_cols, --use_abs
    """
    p = argparse.ArgumentParser(
        description="CLUE/L1000 level3 gctx -> HGNC-token shards (server-friendly)."
    )

    # === Input ===
    p.add_argument("--gctx", type=str, required=True,
                   help="Path to level3 gctx file")
    p.add_argument("--conditional", type=str, required=True,
                   help="Path to siginfo/conditional table (TSV)")
    p.add_argument("--cellinfo", type=str, required=True,
                   help="Path to cellinfo table (TSV)")
    p.add_argument("--compoundinfo", type=str, required=True,
                   help="Path to compoundinfo table (TSV)")
    p.add_argument("--geneinfo", type=str, required=True,
                   help="Path to geneinfo table (TSV, Entrez → Ensembl mapping)")

    # === HGNC Gene Mapping ===
    p.add_argument("--hgnc", type=str, required=True,
                   help="Path to hgnc_complete_set.txt (tab-separated)")

    # === Output ===
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory")

    # === Hyperparameters ===
    p.add_argument("--topk", type=int, default=1024,
                   help="Top-K genes per signature = sequence length (default: 1024)")
    p.add_argument("--n_bins", type=int, default=32,
                   help="Number of bins for rank-based binning (default: 32)")
    p.add_argument("--chunk_rows", type=int, default=20000,
                   help="Chunk size for streaming gctx (default: 20000)")
    p.add_argument("--shard_size", type=int, default=50000,
                   help="Number of signatures per shard (default: 50000)")
    p.add_argument("--extra", type=int, default=128,
                   help="Overselect TOPK+EXTRA to handle unmapped genes (default: 128)")
    p.add_argument("--use_abs", action="store_true",
                   help="Use abs(expression) for gene ranking (recommended for level3)")

    # === Metadata ===
    p.add_argument("--text_cols", type=str,
                   default="tissue,region,sampleType,compartment,cellType1,cellType2,"
                           "majorCluster,subCluster,sex,age,perturbType,disease,drug,dose,time",
                   help="Comma-separated metadata columns to save as text_data "
                        "(aligned with single-cell pipeline)")

    return p.parse_args()


def main():
    """
    Main entry point: execute processing steps sequentially.

    Flow:
        Step 0: Load metadata (conditional + cellinfo + compoundinfo)
        Step 1: Build gene mapping (Entrez → Ensembl → HGNC numeric ID) + vocabulary
        Step 2: Tokenize and shard
        Step 3: Save domain mapping
    """
    args = parse_args()
    gctx_path = Path(args.gctx)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # =========================================================================
    # Step 0: Load Metadata
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Step 0] Loading metadata ...")
    print("=" * 60)
    cond_keep, _pertid_to_cmap = step_load_metadata(
        conditional_path=Path(args.conditional),
        cellinfo_path=Path(args.cellinfo),
        compoundinfo_path=Path(args.compoundinfo),
    )

    # =========================================================================
    # Step 1: Build Gene Mapping + Vocabulary
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Step 1] Building gene mapping + vocabulary ...")
    print("=" * 60)

    # Read gctx gene IDs (Entrez) and save reference files
    with h5py.File(str(gctx_path), "r") as f:
        rid_ds = f["/0/META/ROW/id"]
        gctx_entrez = [x.decode("utf-8") for x in rid_ds[:]]

    # Save Entrez gene list
    (out_dir / "genes_entrez.json").write_text(json.dumps(gctx_entrez))
    print(f"[REF] Saved genes_entrez.json ({len(gctx_entrez)} genes)")

    # Save gene symbols (for interpretability)
    gi_sym = pd.read_csv(args.geneinfo, sep="\t", dtype=str,
                         usecols=["gene_id", "gene_symbol"])
    mp_sym = dict(zip(gi_sym["gene_id"].astype(str),
                      gi_sym["gene_symbol"].astype(str)))
    genes_symbol = [mp_sym.get(e, "") for e in gctx_entrez]
    (out_dir / "genes_symbol.json").write_text(json.dumps(genes_symbol))
    print(f"[REF] Saved genes_symbol.json")

    # Build mapping
    map_g_to_hgnctoken, _vocab = step_build_gene_mapping(
        gctx_entrez=gctx_entrez,
        geneinfo_path=Path(args.geneinfo),
        hgnc_path=Path(args.hgnc),
        out_dir=out_dir,
    )

    # =========================================================================
    # Step 2: Tokenize + Shard
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Step 2] Tokenizing and writing shards ...")
    print("=" * 60)
    domain_map = step_tokenize_shards(
        gctx_path=gctx_path,
        out_dir=out_dir,
        map_g_to_hgnctoken=map_g_to_hgnctoken,
        cond_keep=cond_keep,
        topk=args.topk,
        n_bins=args.n_bins,
        extra=args.extra,
        chunk_rows=args.chunk_rows,
        shard_size=args.shard_size,
        use_abs=args.use_abs,
        text_cols=args.text_cols,
    )

    # =========================================================================
    # Step 3: Save Domain Mapping
    # =========================================================================
    (out_dir / "domain_map.json").write_text(json.dumps(domain_map, indent=2))
    print(f"\n[DOMAIN] Saved domain_map.json (domains={len(domain_map):,})")

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {out_dir / 'gene_vocab.json'}                      <- Gene vocab (HGNC numeric ID -> token_id)")
    print(f"  - {out_dir / 'conn_id_to_hgnc.json'}                <- HGNC numeric ID -> gene symbol")
    print(f"  - {out_dir / 'gene_map_gctxindex_to_hgnctoken.npy'} <- gctx index -> token_id")
    print(f"  - {out_dir / 'gctx_gene_to_hgncid.tsv'}             <- Debug mapping table")
    print(f"  - {out_dir / 'genes_entrez.json'}                    <- Entrez gene list")
    print(f"  - {out_dir / 'genes_symbol.json'}                    <- Gene symbols")
    print(f"  - {out_dir / 'domain_map.json'}                      <- Domain mapping")
    print(f"  - {out_dir / 'shards/'}                              <- Tokenized data shards")


if __name__ == "__main__":
    main()
