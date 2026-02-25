"""
=============================================================================
Server-friendly preprocessing for Tahoe-100M (HF Arrow dataset on disk)
Output format: gene-token sequence + rank-based binned expression (+ metadata)
=============================================================================

This script mirrors the design/outputs of `data_process_L1000_v0.1.py`.

=============================================================================
Pipeline overview (3 core steps)
=============================================================================

Step 0: Load metadata
  - Load obs_metadata (filter pass_filter == "full"), select needed columns
  - Merge cell_line_metadata (cell_name → Organ)
  - Build dose string from drugname_drugconc
  - Create standardized metadata columns matching the single-cell pipeline:
      tissue, region, sampleType, compartment, cellType1, cellType2,
      majorCluster, subCluster, sex, age, perturbType, disease, drug, dose, time
  - Index result by BARCODE_SUB_LIB_ID

Step 1: Build gene vocabulary
  - Load gene_metadata (gene_symbol, ensembl_id, token_id)
  - Load hgnc_complete_set.txt
  - Mapping chain (primary):  ensembl_id (gene_metadata) → ensembl_gene_id (HGNC) → HGNC numeric ID
  - Mapping chain (fallback): gene_symbol (gene_metadata) → symbol/alias (HGNC) → HGNC numeric ID
  - Build vocabulary (dict format, aligned with single-cell / L1000 pipeline):
      {"<pad>": 0, "<cls>": 1, "5": 2, ...}  (keys are HGNC numeric ID strings)
  - Build mapping array: original token_id → new HGNC token_id (for fast lookup during Step 2)
  - Outputs:
      * gene_vocab.json              (HGNC numeric ID → token_id)
      * conn_id_to_hgnc.json        (HGNC numeric ID → gene_symbol, for interpretation)
      * gene_token_to_hgncid.tsv    (debug: original token_id → HGNC numeric ID)

Step 2: Tokenize + shard
  - Stream expression_data in batches
  - Filter: only rows whose BARCODE_SUB_LIB_ID is in obs_keep (QC-passed)
  - For each observation:
      1) Drop invalid token ids (<=0)
      2) Select Top-K genes by abs(expression) (or expression if not use_abs)
      3) Rank-based binning: bin = n_bins - (rank * n_bins) // L
         rank=0 (highest) → bin=n_bins; rank=L-1 (lowest) → bin=1; padding=0
      4) Pad to K with 0
  - Outputs:
      * shards/shard_XXXXX.npz
      * domain_map.json

Each shard contains:
  gene_ids  int32   [B, K]   - token ids (0 = <pad>)
  bin_ids   int16   [B, K]   - rank-based bins (0 = padding, 1~n_bins = real)
  lengths   int16   [B]      - non-PAD count per row
  text_data U       [B, C]   - standard metadata columns (unicode)
  text_cols U       [C]      - column names
  domain_id int32   [B]      - coarse domain id
  sample_id U       [B]      - BARCODE_SUB_LIB_ID

=============================================================================
Design notes
=============================================================================

Token space:
  - Vocabulary keys are HGNC numeric IDs (e.g., "5" from "HGNC:5"), identical to
    the single-cell pipeline (data_process_v0.0.py) and L1000 pipeline
    (data_process_L1000_v0.1.py). This allows multi-modal pretraining where Tahoe,
    L1000 signatures, and single-cell data share the same gene token space.
  - Tahoe gene_metadata provides ensembl_id + gene_symbol; these are mapped to HGNC
    numeric IDs via hgnc_complete_set.txt (Ensembl primary, symbol fallback).
  - A mapping array (original token_id → new HGNC token_id) is built in Step 1 and
    applied during Step 2 tokenization.

Text data alignment:
  - Same 15 standard columns as L1000 and single-cell pipelines.
  - cellType1: cell_name (cancer cell line name).
  - tissue: Organ from cell_line_metadata (via cell_name).
  - dose: parsed from obs_metadata.drugname_drugconc (e.g. "0.05 uM").
  - Fields not available in Tahoe are filled with "none".

QC:
  - Observations filtered by obs_metadata.pass_filter == "full" (configurable
    via --qc_col / --qc_val).
  - Memory note: obs_metadata has ~101M rows. Loading is done via HF datasets
    select_columns + filter before to_pandas() to reduce peak memory.

Domain id:
  - coarse domain string: plate|sublibrary (from obs_metadata)

=============================================================================
Example
=============================================================================

python data_process_Tahoe.py \
  --dataset_dir /path/to/Tahoe-100M \
  --hgnc hgnc_complete_set.txt \
  --out_dir out_Tahoe \
  --topk 1024 --n_bins 32 --batch_size 4096 --shard_size 50000 --use_abs

=============================================================================
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from datasets import load_from_disk
except Exception as e:
    raise RuntimeError(
        "Missing dependency `datasets`. Install with: pip install datasets"
    ) from e


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
    """Convert value to string; treat NaN/None/empty/null as 'none'."""
    if x is None:
        return "none"
    s = str(x)
    if s == "" or s.lower() in ("nan", "none", "na", "null"):
        return "none"
    return s


def as_list(x):
    """Normalize to python list (HF Datasets may return list/ndarray)."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return list(x)
    except Exception:
        return [x]


def parse_drugname_drugconc(val) -> str:
    """
    Extract dose string from obs_metadata.drugname_drugconc.

    Input format: "[('Drug name', 0.05, 'uM')]"
    Output: "0.05 uM"  (multiple entries joined by ', ')
    Falls back to the raw string if parsing fails.
    """
    s = safe_str(val)
    if s == "none":
        return "none"
    try:
        data = ast.literal_eval(s)
        if isinstance(data, list) and data:
            parts = []
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    parts.append(f"{entry[1]} {entry[2]}")
            if parts:
                return ", ".join(parts)
    except Exception:
        pass
    return s


# =============================================================================
# Metadata helpers
# =============================================================================

STD_TEXT_COLS_DEFAULT = (
    "tissue,region,sampleType,compartment,cellType1,cellType2,"
    "majorCluster,subCluster,sex,age,perturbType,disease,drug,dose,time"
)


def coarse_domain_str(plate: str, sublibrary: str) -> str:
    """Coarse domain key: plate|sublibrary."""
    return "|".join([safe_str(plate), safe_str(sublibrary)])


# =============================================================================
# Step 0: Load and merge metadata
# =============================================================================

def step_load_metadata(
    dataset_dir: Path,
    subset_split: str,
    qc_col: str,
    qc_val: str,
) -> pd.DataFrame:
    """
    Step 0: Load obs_metadata (QC filter) + cell_line_metadata, build
    standardized 15-column metadata indexed by BARCODE_SUB_LIB_ID.

    Process:
      1. Load cell_line_metadata; build cell_name → Organ lookup
      2. Load obs_metadata with only needed columns; filter qc_col == qc_val
      3. Compute standardized columns:
           tissue     ← Organ from cell_line_metadata (via cell_name)
           cellType1  ← cell_name
           drug       ← obs_metadata.drug
           dose       ← parsed from obs_metadata.drugname_drugconc
           perturbType← "drug"
           others     ← "none"
      4. Keep plate + sublibrary (needed for domain id)
      5. Index by BARCODE_SUB_LIB_ID

    Args:
        dataset_dir: Root directory of Tahoe-100M dataset
        subset_split: HF split name (e.g., "train")
        qc_col: Column in obs_metadata used for QC filtering (default: "pass_filter")
        qc_val: Value that passes QC (default: "full")

    Returns:
        obs_keep: DataFrame indexed by BARCODE_SUB_LIB_ID with standardized columns
    """
    # ---- Load cell_line_metadata (small, ~1k rows) ----
    cell_ds = load_from_disk(str(dataset_dir / "cell_line_metadata"))[subset_split]
    cell_df = cell_ds.to_pandas()
    print(f"[META] cell_line_metadata rows: {len(cell_df):,}")

    name_to_organ: Dict[str, str] = {}
    if "cell_name" in cell_df.columns and "Organ" in cell_df.columns:
        name_to_organ = {
            safe_str(k): safe_str(v)
            for k, v in zip(cell_df["cell_name"].astype(str),
                            cell_df["Organ"].astype(str))
        }
    print(f"[META] cell_name→Organ lookup: {len(name_to_organ):,} entries")

    # ---- Load obs_metadata (large, ~101M rows): select needed columns only ----
    obs_ds = load_from_disk(str(dataset_dir / "obs_metadata"))[subset_split]
    print(f"[META] obs_metadata total rows: {obs_ds.num_rows:,}")

    if qc_col not in obs_ds.column_names:
        raise ValueError(
            f"obs_metadata missing QC column '{qc_col}'. "
            f"Available: {obs_ds.column_names}"
        )

    obs_needed = [
        "BARCODE_SUB_LIB_ID", qc_col,
        "sublibrary", "plate",
        "cell_name", "drug", "drugname_drugconc",
    ]
    obs_avail   = [c for c in obs_needed if c in obs_ds.column_names]
    obs_missing = [c for c in obs_needed if c not in obs_ds.column_names]
    if obs_missing:
        print(f"[META] obs_metadata missing columns (will fill 'none'): {obs_missing}")

    print(f"[META] Filtering {qc_col} == '{qc_val}' ...")
    obs_filtered = (
        obs_ds
        .select_columns(obs_avail)
        .filter(lambda batch: [v == qc_val for v in batch[qc_col]], batched=True)
    )
    obs_df = obs_filtered.to_pandas()
    print(f"[META] obs_metadata after QC ({qc_col}=={qc_val!r}): {len(obs_df):,} rows")

    # ---- Standardized columns ----
    if "cell_name" in obs_df.columns:
        obs_df["tissue"]   = obs_df["cell_name"].apply(
            lambda cn: name_to_organ.get(safe_str(cn), "none")
        )
        obs_df["cellType1"] = obs_df["cell_name"].apply(safe_str)
    else:
        obs_df["tissue"]    = "none"
        obs_df["cellType1"] = "none"

    obs_df["drug"] = obs_df["drug"].apply(safe_str) if "drug" in obs_df.columns else "none"
    obs_df["dose"] = (
        obs_df["drugname_drugconc"].apply(parse_drugname_drugconc)
        if "drugname_drugconc" in obs_df.columns
        else "none"
    )

    # Not available in Tahoe:
    for col in ("region", "sampleType", "compartment", "cellType2",
                "majorCluster", "subCluster", "sex", "age", "disease", "time"):
        obs_df[col] = "none"
    obs_df["perturbType"] = "drug"

    # ---- Keep only needed columns; index by BARCODE_SUB_LIB_ID ----
    if "BARCODE_SUB_LIB_ID" not in obs_df.columns:
        raise ValueError("obs_metadata is missing 'BARCODE_SUB_LIB_ID' column.")

    final_cols = [
        "plate", "sublibrary",
        "tissue", "region", "sampleType", "compartment",
        "cellType1", "cellType2", "majorCluster", "subCluster",
        "sex", "age", "perturbType", "disease", "drug", "dose", "time",
    ]
    final_cols_avail = [c for c in final_cols if c in obs_df.columns]
    obs_keep = obs_df[final_cols_avail].copy()
    obs_keep.index = obs_df["BARCODE_SUB_LIB_ID"].astype(str)
    obs_keep.index.name = "BARCODE_SUB_LIB_ID"

    return obs_keep


# =============================================================================
# Step 1: Build gene vocabulary
# =============================================================================

def step_build_gene_vocab(
    dataset_dir: Path,
    subset_split: str,
    hgnc_path: Path,
    out_dir: Path,
) -> np.ndarray:
    """
    Step 1: Build HGNC-based gene vocabulary from gene_metadata + hgnc_complete_set.txt.

    Mapping chain (primary):
        gene_metadata.ensembl_id → (HGNC table) → HGNC numeric ID

    Fallback (when Ensembl mapping fails):
        gene_metadata.gene_symbol → (HGNC symbol/alias/prev_symbol) → HGNC numeric ID

    Vocabulary format (aligned with data_process_v0.0.py and L1000 pipeline):
        {"<pad>": 0, "<cls>": 1, "5": 2, "7": 3, ...}
        (keys are HGNC numeric ID strings extracted from "HGNC:5")

    Args:
        dataset_dir: Root directory of Tahoe-100M dataset
        subset_split: HF split name (e.g., "train")
        hgnc_path: Path to hgnc_complete_set.txt (tab-separated)
        out_dir: Output directory

    Returns:
        orig_to_hgnctoken: int32 array of length max_orig_token_id+1,
            where orig_to_hgnctoken[orig_token_id] = new_hgnc_token_id (0=unmapped)

    Outputs:
        gene_vocab.json           - HGNC numeric ID → token_id (dict)
        conn_id_to_hgnc.json      - HGNC numeric ID → gene symbol (for interpretation)
        gene_token_to_hgncid.tsv  - Debug: original token_id → HGNC numeric ID
    """
    # ---- Load gene_metadata ----
    gene_ds = load_from_disk(str(dataset_dir / "gene_metadata"))[subset_split]
    gene_df = gene_ds.to_pandas()
    print(f"[VOCAB] gene_metadata rows: {len(gene_df):,}")

    gene_df = gene_df.dropna(subset=["ensembl_id", "token_id"]).copy()
    gene_df["ensembl_norm"]  = gene_df["ensembl_id"].astype(str).apply(norm_ensembl)
    gene_df["orig_token_id"] = gene_df["token_id"].astype(int)

    # ---- Load HGNC ----
    hgnc_df = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    print(f"[HGNC] Loaded {len(hgnc_df):,} rows from {hgnc_path.name}")

    # Extract HGNC numeric ID: "HGNC:5" → "5"
    hgnc_df["hgnc_num"] = hgnc_df["hgnc_id"].apply(
        lambda x: str(x).strip().split(":", 1)[-1]
        if pd.notna(x) and ":" in str(x) else ""
    )

    # ---- Primary lookup: Ensembl → HGNC numeric ID ----
    hgnc_df["ensembl_norm"] = hgnc_df["ensembl_gene_id"].apply(norm_ensembl)
    hgnc_ens = hgnc_df.loc[
        (hgnc_df["ensembl_norm"] != "") & (hgnc_df["hgnc_num"] != ""),
        ["ensembl_norm", "hgnc_num"]
    ].drop_duplicates("ensembl_norm", keep="first")

    # ---- Fallback lookup: symbol/alias/prev_symbol → HGNC numeric ID ----
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

    # ---- Join gene_metadata with HGNC by Ensembl ID ----
    gene_df2 = gene_df[["orig_token_id", "ensembl_norm", "gene_symbol"]].copy()
    gene_df2 = gene_df2.merge(hgnc_ens, on="ensembl_norm", how="left")

    # ---- Symbol fallback ----
    if symbol_to_hgnc_num:
        primary_failed = (
            gene_df2["hgnc_num"].isna() |
            gene_df2["hgnc_num"].str.lower().isin(["nan", "none", ""])
        )
        if primary_failed.any():
            gene_df2["_sym"] = gene_df2["gene_symbol"].apply(
                lambda x: str(x).strip()
                if pd.notna(x) and str(x).strip().lower() not in ("nan", "none", "") else ""
            )
            can_fallback = primary_failed & (gene_df2["_sym"] != "")
            gene_df2.loc[can_fallback, "hgnc_num"] = (
                gene_df2.loc[can_fallback, "_sym"].map(symbol_to_hgnc_num)
            )
            gene_df2.drop(columns=["_sym"], inplace=True)

            still_failed = (
                gene_df2["hgnc_num"].isna() |
                gene_df2["hgnc_num"].str.lower().isin(["nan", "none", ""])
            )
            n_ensembl = (~primary_failed).sum()
            n_symbol  = (primary_failed & ~still_failed).sum()
            print(f"[GENE MAP] Mapped via Ensembl: {n_ensembl:,}, "
                  f"via symbol fallback: {n_symbol:,}, "
                  f"still unmapped: {still_failed.sum():,}")

    # ---- Build vocabulary (HGNC numeric ID strings, sorted numerically) ----
    valid_mask = (
        gene_df2["hgnc_num"].notna() &
        ~gene_df2["hgnc_num"].str.lower().isin(["nan", "none", ""])
    )
    uniq_hgnc_ids = sorted(
        set(gene_df2.loc[valid_mask, "hgnc_num"]),
        key=lambda x: int(x) if x.isdigit() else 0
    )

    vocab: Dict[str, int] = {"<pad>": 0, "<cls>": 1}
    for hid in uniq_hgnc_ids:
        vocab[hid] = len(vocab)

    (out_dir / "gene_vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"[VOCAB] Saved gene_vocab.json (size={len(vocab):,}) -> {out_dir / 'gene_vocab.json'}")

    # ---- Build conn_id_to_hgnc (HGNC numeric ID → gene symbol) ----
    conn_to_hgnc: Dict[str, str] = {"<pad>": "<pad>", "<cls>": "<cls>"}
    for _, row in gene_df2[valid_mask].iterrows():
        hid = str(row["hgnc_num"])
        sym = str(row["gene_symbol"]) if pd.notna(row.get("gene_symbol")) else ""
        if hid in vocab and hid not in conn_to_hgnc:
            conn_to_hgnc[hid] = sym

    (out_dir / "conn_id_to_hgnc.json").write_text(json.dumps(conn_to_hgnc, indent=2))
    print(f"[VOCAB] Saved conn_id_to_hgnc.json -> {out_dir / 'conn_id_to_hgnc.json'}")

    # ---- Build orig_token_id → hgnc_token_id mapping array ----
    # Array index = original token_id, value = new HGNC token_id (0 = unmapped)
    max_orig = int(gene_df2["orig_token_id"].max())
    orig_to_hgnctoken = np.zeros(max_orig + 1, dtype=np.int32)

    mapped_df = gene_df2[valid_mask].copy()
    mapped_df["hgnc_token"] = mapped_df["hgnc_num"].map(vocab).fillna(0).astype(int)
    for row in mapped_df[["orig_token_id", "hgnc_token"]].itertuples(index=False):
        orig_to_hgnctoken[int(row.orig_token_id)] = int(row.hgnc_token)

    # ---- Debug table ----
    gene_df2["hgnc_token"] = gene_df2["orig_token_id"].apply(
        lambda t: int(orig_to_hgnctoken[int(t)]) if 0 <= int(t) <= max_orig else 0
    )
    gene_df2.to_csv(out_dir / "gene_token_to_hgncid.tsv", sep="\t", index=False)
    print(f"[GENE MAP] Total genes: {len(gene_df2):,}, mapped: {valid_mask.sum():,}")
    print(f"[GENE MAP] Saved debug table -> gene_token_to_hgncid.tsv")

    return orig_to_hgnctoken


# =============================================================================
# Shard writer
# =============================================================================

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

    Uses unicode dtype ("U") for string data, matching the single-cell and
    L1000 pipeline conventions.
    """
    gene_ids = np.concatenate(buf_gene, axis=0) if buf_gene else np.zeros((0, 0), dtype=np.int32)
    bin_ids  = np.concatenate(buf_bin,  axis=0) if buf_bin  else np.zeros((0, 0), dtype=np.int16)
    lengths  = np.concatenate(buf_len,  axis=0) if buf_len  else np.zeros((0,),   dtype=np.int16)

    payload = dict(
        gene_ids=gene_ids,
        bin_ids=bin_ids,
        lengths=lengths,
        domain_id=np.array(buf_domain, dtype=np.int32),
        sample_id=np.asarray(buf_sid, dtype="U"),
    )

    if text_cols_list is not None and buf_text:
        payload["text_data"] = np.concatenate(buf_text, axis=0)
        payload["text_cols"] = np.asarray(text_cols_list, dtype="U")

    np.savez_compressed(str(path), **payload)
    print(f"[SHARD] {path.name}  n={gene_ids.shape[0]:,}")


# =============================================================================
# Core tokenization (per observation)
# =============================================================================

def tokenize_one(
    genes: List[int],
    exprs: List[float],
    topk: int,
    n_bins: int,
    use_abs: bool,
    orig_to_hgnctoken: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Tokenize one observation.

    Applies orig_to_hgnctoken mapping to convert original token_ids (from
    gene_metadata) to HGNC-based token_ids before selection and binning.

    Binning convention (matching v0.0 / L1000):
      bin = n_bins - (rank * n_bins) // L
      rank 0 (highest expression) → bin n_bins
      rank L-1 (lowest expression) → bin 1
      padding positions remain 0

    Args:
        genes: Original token_ids from expression_data
        exprs: Expression values
        topk: Max genes per observation
        n_bins: Number of expression bins
        use_abs: Rank by abs(expression) if True
        orig_to_hgnctoken: int32 array, index=orig_token_id, value=hgnc_token_id (0=unmapped)

    Returns:
        gene_ids[topk], bin_ids[topk], length
    """
    g = np.asarray(genes, dtype=np.int64)
    x = np.asarray(exprs, dtype=np.float32)

    if g.size == 0 or x.size == 0:
        return np.zeros((topk,), dtype=np.int32), np.zeros((topk,), dtype=np.int16), 0

    L0 = min(g.size, x.size)
    g, x = g[:L0], x[:L0]

    # Map original token_ids → HGNC token_ids (vectorized)
    n_map = len(orig_to_hgnctoken)
    in_range = (g >= 0) & (g < n_map)
    g_hgnc = np.where(in_range, orig_to_hgnctoken[np.clip(g, 0, n_map - 1)], 0)

    # Drop unmapped (0 = <pad> / unmapped)
    keep = g_hgnc > 0
    g, x = g_hgnc[keep].astype(np.int64), x[keep]

    if g.size == 0:
        return np.zeros((topk,), dtype=np.int32), np.zeros((topk,), dtype=np.int16), 0

    score = np.abs(x) if use_abs else x

    # Top-K selection + descending sort
    K0 = min(topk, g.size)
    if g.size > K0:
        idx = np.argpartition(-score, K0 - 1)[:K0]
        idx = idx[np.argsort(-score[idx])]
    else:
        idx = np.argsort(-score)
    g = g[idx]

    L = min(topk, g.size)
    out_gene = np.zeros((topk,), dtype=np.int32)
    out_bin  = np.zeros((topk,), dtype=np.int16)
    out_gene[:L] = g[:L].astype(np.int32)

    ranks = np.arange(L, dtype=np.int32)
    out_bin[:L] = (n_bins - (ranks * n_bins) // max(L, 1)).astype(np.int16)

    return out_gene, out_bin, int(L)


# =============================================================================
# Step 2: Tokenize + shard
# =============================================================================

def step_tokenize_shards(
    dataset_dir: Path,
    out_dir: Path,
    obs_keep: pd.DataFrame,
    orig_to_hgnctoken: np.ndarray,
    topk: int,
    n_bins: int,
    batch_size: int,
    shard_size: int,
    use_abs: bool,
    text_cols_list: List[str],
    subset_split: str,
    expression_subset: str,
) -> Dict[str, int]:
    """
    Step 2: Stream expression_data, filter to QC-passed observations, tokenize,
    and write shards.

    Mirrors L1000's step_tokenize_shards:
      - obs_keep (indexed by BARCODE_SUB_LIB_ID) replaces cond_keep (indexed by sample_id)
      - expression_data streaming replaces gctx chunk reading
      - orig_to_hgnctoken maps original token_ids → HGNC token_ids (analogous to
        gene_map_gctxindex_to_hgnctoken.npy in L1000)

    Args:
        dataset_dir: Root Tahoe-100M directory
        out_dir: Output directory
        obs_keep: DataFrame indexed by BARCODE_SUB_LIB_ID (from step_load_metadata)
        orig_to_hgnctoken: int32 array, index=orig_token_id, value=hgnc_token_id (0=unmapped)
        topk: Max genes per observation = sequence length
        n_bins: Number of rank-based expression bins
        batch_size: HF streaming batch size
        shard_size: Observations per output shard
        use_abs: Rank by abs(expression) if True
        text_cols_list: Ordered list of metadata column names to save
        subset_split: HF split name
        expression_subset: Subfolder name for expression data

    Returns:
        domain_map: {domain_string: domain_id}
    """
    shards_dir = out_dir / "shards"
    ensure_dir(shards_dir)

    expr_ds = load_from_disk(str(dataset_dir / expression_subset))[subset_split]
    print(f"[GCTX] expression_data rows: {expr_ds.num_rows:,}")

    # Warn about any text columns missing from obs_keep
    text_cols_missing = [c for c in text_cols_list if c not in obs_keep.columns]
    if text_cols_missing:
        print(f"[TOK] Warning: missing text columns (will fill 'none'): {text_cols_missing}")
    print(f"[TOK] Will save text_data columns: {text_cols_list}")

    global_domain_to_id: Dict[str, int] = {}
    shard_id  = 0
    buf_gene:   List[np.ndarray] = []
    buf_bin:    List[np.ndarray] = []
    buf_len:    List[np.ndarray] = []
    buf_text:   List[np.ndarray] = []
    buf_domain: List[int]        = []
    buf_sid:    List[str]        = []
    buf_count   = 0

    iterable = expr_ds.to_iterable_dataset(num_shards=64).iter(batch_size=batch_size)
    pbar = tqdm(iterable, desc="[TOK] chunking expression_data")

    for batch in pbar:
        bsl_batch   = batch.get("BARCODE_SUB_LIB_ID", None)
        genes_batch = batch.get("genes", None)
        exprs_batch = batch.get("expressions", None)

        if genes_batch is None or exprs_batch is None:
            raise ValueError("expression_data must contain columns: 'genes' and 'expressions'")
        if bsl_batch is None:
            raise ValueError("expression_data must contain column: 'BARCODE_SUB_LIB_ID'")

        # ---- Filter: only keep QC-passed observations ----
        idx_pd  = pd.Index(bsl_batch)
        in_mask = idx_pd.isin(obs_keep.index)
        if not in_mask.any():
            continue

        keep_pos = np.where(in_mask.to_numpy())[0]
        keep_ids = [bsl_batch[i] for i in keep_pos]

        meta = obs_keep.loc[keep_ids].copy()
        meta = meta.reindex(keep_ids)

        B = len(keep_pos)
        out_gene = np.zeros((B, topk), dtype=np.int32)
        out_bin  = np.zeros((B, topk), dtype=np.int16)
        out_len  = np.zeros((B,), dtype=np.int16)

        for bi, orig_i in enumerate(keep_pos):
            g_list = as_list(genes_batch[orig_i])
            x_list = as_list(exprs_batch[orig_i])
            g_ids, b_ids, L = tokenize_one(
                g_list, x_list, topk=topk, n_bins=n_bins, use_abs=use_abs,
                orig_to_hgnctoken=orig_to_hgnctoken,
            )
            out_gene[bi] = g_ids
            out_bin[bi]  = b_ids
            out_len[bi]  = L

        buf_gene.append(out_gene)
        buf_bin.append(out_bin)
        buf_len.append(out_len)

        # ---- Build text_data (2D unicode array, matching L1000/v0.0 format) ----
        text_arr = np.empty((B, len(text_cols_list)), dtype="U")
        for j, col in enumerate(text_cols_list):
            if col in meta.columns:
                text_arr[:, j] = meta[col].apply(safe_str).values
            else:
                text_arr[:, j] = "none"
        buf_text.append(text_arr)

        # ---- Domain IDs + sample IDs ----
        for bi in range(B):
            row = meta.iloc[bi]
            plate      = safe_str(row.get("plate",      "none"))
            sublibrary = safe_str(row.get("sublibrary", "none"))
            dom = coarse_domain_str(plate=plate, sublibrary=sublibrary)
            if dom not in global_domain_to_id:
                global_domain_to_id[dom] = len(global_domain_to_id)
            buf_domain.append(global_domain_to_id[dom])
            buf_sid.append(keep_ids[bi])

        buf_count += B

        # ---- Write shard if buffer is full ----
        if buf_count >= shard_size:
            _write_shard(
                shards_dir / f"shard_{shard_id:05d}.npz",
                buf_gene, buf_bin, buf_len, buf_text,
                buf_domain, buf_sid, text_cols_list,
            )
            shard_id += 1
            buf_gene, buf_bin, buf_len, buf_text = [], [], [], []
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


# =============================================================================
# Command Line Interface (CLI)
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Argument groups:
        Input: --dataset_dir, --subset_split, --expression_subset
        QC: --qc_col, --qc_val
        Output: --out_dir
        Hyperparameters: --topk, --n_bins, --batch_size, --shard_size, --use_abs
        Metadata: --text_cols
    """
    p = argparse.ArgumentParser(
        description="Tahoe-100M -> gene-token shards (server-friendly, mirrors L1000 pipeline)."
    )

    # === Input ===
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Path to Tahoe-100M dataset directory (contains expression_data/, etc.)")
    p.add_argument("--subset_split", type=str, default="train",
                   help="HF split name (default: train)")
    p.add_argument("--expression_subset", type=str, default="expression_data",
                   help="Expression data subfolder name (default: expression_data)")

    # === HGNC Gene Mapping ===
    p.add_argument("--hgnc", type=str, required=True,
                   help="Path to hgnc_complete_set.txt (tab-separated)")

    # === QC ===
    p.add_argument("--qc_col", type=str, default="pass_filter",
                   help="obs_metadata column for QC filtering (default: pass_filter)")
    p.add_argument("--qc_val", type=str, default="full",
                   help="QC pass value (default: full)")

    # === Output ===
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory")

    # === Hyperparameters ===
    p.add_argument("--topk", type=int, default=1024,
                   help="Top-K genes per observation = sequence length (default: 1024)")
    p.add_argument("--n_bins", type=int, default=32,
                   help="Number of rank-based expression bins (default: 32)")
    p.add_argument("--batch_size", type=int, default=4096,
                   help="Streaming batch size (default: 4096)")
    p.add_argument("--shard_size", type=int, default=50000,
                   help="Observations per output shard (default: 50000)")
    p.add_argument("--use_abs", action="store_true",
                   help="Rank by abs(expression) for gene selection (recommended)")

    # === Metadata ===
    p.add_argument("--text_cols", type=str, default=STD_TEXT_COLS_DEFAULT,
                   help="Comma-separated metadata columns to store as text_data "
                        "(aligned with L1000 and single-cell pipelines)")

    return p.parse_args()


def main():
    """
    Main entry point: execute processing steps sequentially.

    Flow:
        Step 0: Load metadata (obs_metadata + cell_line_metadata)
        Step 1: Build gene vocabulary (from gene_metadata)
        Step 2: Tokenize expression_data and write shards
        Step 3: Save domain mapping
    """
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)
    ensure_dir(out_dir)

    text_cols_list = [c.strip() for c in args.text_cols.split(",") if c.strip()]

    # =========================================================================
    # Step 0: Load Metadata
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Step 0] Loading metadata ...")
    print("=" * 60)
    obs_keep = step_load_metadata(
        dataset_dir=dataset_dir,
        subset_split=args.subset_split,
        qc_col=args.qc_col,
        qc_val=args.qc_val,
    )
    print(f"[META] obs_keep: {len(obs_keep):,} observations, {len(obs_keep.columns)} columns")
    print(f"[META] text_data columns: {text_cols_list}")

    # =========================================================================
    # Step 1: Build Gene Vocabulary
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Step 1] Building gene vocabulary ...")
    print("=" * 60)
    orig_to_hgnctoken = step_build_gene_vocab(
        dataset_dir=dataset_dir,
        subset_split=args.subset_split,
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
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        obs_keep=obs_keep,
        orig_to_hgnctoken=orig_to_hgnctoken,
        topk=args.topk,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        use_abs=args.use_abs,
        text_cols_list=text_cols_list,
        subset_split=args.subset_split,
        expression_subset=args.expression_subset,
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
    print(f"  - {out_dir / 'gene_vocab.json'}             <- HGNC numeric ID → token_id")
    print(f"  - {out_dir / 'conn_id_to_hgnc.json'}        <- HGNC numeric ID → gene symbol")
    print(f"  - {out_dir / 'gene_token_to_hgncid.tsv'}    <- Debug: original token_id → HGNC numeric ID")
    print(f"  - {out_dir / 'domain_map.json'}              <- domain mapping")
    print(f"  - {out_dir / 'shards/'}                      <- tokenized data shards")


if __name__ == "__main__":
    main()
