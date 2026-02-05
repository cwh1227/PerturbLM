"""
=============================================================================
Server-friendly preprocessing for huge single-cell AnnData (.h5ad)
Output format: gene-token sequence + per-gene binned expression (+ optional metadata)
=============================================================================

Requirements (HPC / server):
  - scanpy, anndata, scipy, numpy, pandas, tqdm

=============================================================================
Pipeline overview (4 core steps)
=============================================================================

Step 1: Cell QC filtering + HVG selection
  - Filters out predicted doublets (obs["predicted_doublet"] == True), if the column exists
  - Performs server-friendly QC on backed AnnData by scanning X in chunks:
      * n_genes  : number of detected genes (non-zero features)
      * counts   : total counts per cell
      * pct_mt   : mitochondrial fraction (genes starting with --mt_prefix)
    and keeps cells passing:
      * min_genes <= n_genes <= max_genes
      * min_counts <= total_counts <= max_counts
      * pct_mt < max_pct_mt
  - Stratified sampling across batches (obs[--batch_key]) to avoid one large batch dominating HVG
  - Runs Scanpy HVG selection using seurat_v3 with batch-aware correction
  - Output:
      * hvg_genes.txt   (original gene names, one per line)

Step 1.5 (Optional): BioMedGraphica gene mapping (normalization + KG alignment)
  - If --biomedgraphica is provided, maps HVG genes to:
      * HGNC symbols
      * BioMedGraphica_Conn_ID (recommended as KG node id)
  - Keeps only genes successfully mapped
  - Outputs:
      * hvg_genes_hgnc.txt        (mapped HGNC symbols)
      * hvg_genes_conn_id.txt     (mapped BioMedGraphica_Conn_ID list)
      * gene_mapping_info.json    (mapping details and rates)

Step 2: Build gene vocabulary (token ids)
  - Builds a token dictionary with special tokens:
      * <pad> = 0
      * <cls> = 1
  - If BioMedGraphica mapping is enabled, the intended design is:
      * vocabulary keys are BioMedGraphica_Conn_ID (so gene_ids align with KG nodes)
    Otherwise:
      * vocabulary keys are original gene names
  - Outputs:
      * gene_vocab.json
      * conn_id_to_hgnc.json
        NOTE: In the current implementation, conn_id_to_hgnc may become an identity mapping
              unless build_gene_vocab(...) is called with the (conn_ids, hgnc) pairs.

Step 3: Per-gene quantile bin edges
  - Loads a sampled subset of cells (after the same QC+doublet filtering)
  - Subsets to HVG genes, normalizes (normalize_total + log1p)
  - For each gene independently, computes quantile edges:
      * n_bins bins => (n_bins - 1) internal quantiles
  - Output:
      * gene_bin_edges.npy with shape [G, n_bins-1] (float32)

Step 4: Tokenize all cells + shard to disk
  - Re-opens h5ad in backed mode, applies the same QC+doublet filtering
  - For each cell:
      1) take non-zero genes
      2) keep Top-K highest expressed genes (K = --topk)
      3) map genes -> token ids
      4) discretize expression -> bin_id via per-gene edges (searchsorted)
      5) sort by expression descending
      6) pad to length K with <pad>=0
  - Writes compressed NPZ shards under out_dir/shards/
  - Each shard contains:
      * gene_ids : int32 [N, K]  - tokenized genes (0 is padding)
      * bin_ids  : int16 [N, K]  - per-gene binned expression
      * lengths  : int16 [N]     - effective lengths before padding
      * text_data: unicode [N, C] (optional; controlled by --text_cols)
      * text_cols: unicode [C]    (optional)
      * domain_id: int32 [N]      (optional; enabled via --save_domain_ids)
    IMPORTANT:
      - Although --target is exposed in CLI, label saving (y) is currently DISABLED
        in Step 4 (the corresponding code is commented out). If you need y,
        re-enable y_all/label_map and payload["y"].

=============================================================================
Design notes
=============================================================================

Pretraining:
  - Usually you do NOT save domain/batch info (model does not take batch input by default)
  - text_data can still be saved if you want a text/metadata encoder later

Downstream:
  - Enable --save_domain_ids to save batch/domain labels for conditional modeling

=============================================================================
Examples
=============================================================================

# Pretrain (no domain_id; text_data saved by default)
python data_process_server.py \
  --h5ad igt_s9_fine_counts.h5ad \
  --out_dir out_pretrain \
  --batch_key datasetID \
  --n_top_genes 4000 \
  --n_bins 32 \
  --topk 1024 \
  --chunk_size 50000

# With BioMedGraphica mapping (recommended)
python data_process_server.py \
  --h5ad igt_s9_fine_counts.h5ad \
  --out_dir out_pretrain_mapped \
  --biomedgraphica BioMedGraphica_Conn_Gene.csv \
  --batch_key datasetID \
  --n_top_genes 4000

# Downstream (save domain_id)
python data_process_server.py \
  --h5ad igt_s9_fine_counts.h5ad \
  --out_dir out_downstream \
  --biomedgraphica BioMedGraphica_Conn_Gene.csv \
  --save_domain_ids \
  --domain_key datasetID
=============================================================================
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def stratified_sample_indices(obs: pd.DataFrame, key: str, n_total: int, seed: int = 0) -> np.ndarray:
    """
    Stratified sampling: Group by obs[key] and sample uniformly from each group.

    Args:
        obs: AnnData obs DataFrame
        key: Grouping column name (e.g., 'datasetID')
        n_total: Total number of samples to collect
        seed: Random seed

    Returns:
        Array of sampled cell indices.

    Example:
        3 datasets with 100k cells each, n_total=6000 -> sample ~2000 from each dataset.
    """
    rng = set_seed(seed)
    if key not in obs.columns:
        raise ValueError(f"batch_key '{key}' not found in obs.columns")
    groups = obs[key].astype(str).fillna("NA")
    uniq = groups.unique().tolist()
    if len(uniq) == 0:
        raise ValueError(f"No groups found in obs['{key}']")

    per = max(1, n_total // len(uniq))
    idx_list = []
    for g in uniq:
        pos = np.where(groups.values == g)[0]
        if pos.size == 0:
            continue
        take = min(per, pos.size)
        idx_list.append(rng.choice(pos, size=take, replace=False))
    if not idx_list:
        raise ValueError("Stratified sampling produced empty index list.")
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    if idx.size > n_total:
        idx = idx[:n_total]
    return idx.astype(np.int64)

def filter_cell_indices_qc_backed(
    adata_b: sc.AnnData,
    filter_doublets: bool = True,
    doublet_col: str = "predicted_doublet",
    # QC thresholds
    min_genes: int = 500,
    max_genes: int = 8000,
    min_counts: int = 1000,
    max_counts: int = 100000,
    max_pct_mt: float = 20.0,
    # mito detection
    mt_prefix: str = "MT-",
    chunk_size: int = 50000,
) -> np.ndarray:
    """
    Perform server-friendly QC filtering on backed AnnData:
      - doublet (obs[doublet_col])
      - n_genes (detected genes per cell, count of non-zero features)
      - total_counts (total counts/UMI per cell)
      - pct_mt (mitochondrial percentage)

    Key points:
      - Does not rely on existing QC columns in obs.
      - Reads X from disk in chunks (adata_b[idx, :]) to calculate metrics and generate keep mask.
    """
    obs = adata_b.obs
    n = adata_b.n_obs

    # 1) Doublet pre-filtering (no X reading)
    keep = np.ones(n, dtype=bool)
    if filter_doublets and (doublet_col in obs.columns):
        keep &= ~(obs[doublet_col].astype(bool).fillna(False).values)

    # 2) Mitochondrial gene indices (no X reading)
    var_names = adata_b.var_names.astype(str)
    mt_mask = var_names.str.startswith(mt_prefix) | var_names.str.startswith(mt_prefix.lower())
    mt_idx = np.where(mt_mask.values)[0].astype(np.int64)
    has_mt = mt_idx.size > 0
    if not has_mt:
        print(f"[QC] Warning: no mitochondrial genes found with prefix '{mt_prefix}'. "
              f"pct_mt will be treated as 0 for all cells.")

    # 3) Chunked X reading to calculate n_genes / total_counts / pct_mt
    keep_idx0 = np.where(keep)[0].astype(np.int64)
    if keep_idx0.size == 0:
        return keep_idx0

    qc_keep = np.zeros(keep_idx0.size, dtype=bool)

    for s in tqdm(range(0, keep_idx0.size, chunk_size), desc="[QC] scanning", unit="cell"):
        e = min(s + chunk_size, keep_idx0.size)
        idx = keep_idx0[s:e]

        X = adata_b[idx, :].X
        if sp.issparse(X):
            X = X.tocsr()
            total_counts = np.asarray(X.sum(axis=1)).reshape(-1)
            n_genes = np.diff(X.indptr)  # non-zeros per row
            if has_mt:
                X_mt = X[:, mt_idx]
                mt_counts = np.asarray(X_mt.sum(axis=1)).reshape(-1)
            else:
                mt_counts = np.zeros_like(total_counts)
        else:
            X = np.asarray(X)
            total_counts = X.sum(axis=1)
            n_genes = (X > 0).sum(axis=1)
            if has_mt:
                mt_counts = X[:, mt_idx].sum(axis=1)
            else:
                mt_counts = np.zeros_like(total_counts)

        pct_mt = np.zeros_like(total_counts, dtype=np.float32)
        nz = total_counts > 0
        pct_mt[nz] = (mt_counts[nz] / total_counts[nz]) * 100.0

        local_keep = (
            (n_genes >= min_genes) & (n_genes <= max_genes) &
            (total_counts >= min_counts) & (total_counts <= max_counts) &
            (pct_mt < max_pct_mt)
        )
        qc_keep[s:e] = local_keep

    kept = keep_idx0[qc_keep]
    print(f"[QC] Kept cells after QC+doublet: {kept.size} / {n}")
    return kept.astype(np.int64)

def normalize_log1p_inplace(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    """
    Standard single-cell preprocessing: normalize_total + log1p (in-place modification).

    Process:
        1. normalize_total: Normalize total counts per cell to target_sum (default 10000).
        2. log1p: Logarithmic transformation log(x+1) to compress range and reduce outlier impact.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

def load_biomedgraphica_mapping(csv_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load BioMedGraphica gene mapping table.

    BioMedGraphica is a biomedical knowledge graph providing standardized gene nomenclature.
    HGNC_Symbol is the official gene symbol from the Human Gene Nomenclature Committee.

    Args:
        csv_path: Path to BioMedGraphica_Conn_Gene.csv

    Returns:
        hgnc_to_conn_id: {HGNC_Symbol: BioMedGraphica_Conn_ID} mapping
        gene_aliases: {Possible Alias: HGNC_Symbol} for matching different formats

    File format:
        BioMedGraphica_Conn_ID, ..., HGNC_Symbol
        BMGC_GN00001, ..., A1BG
    """
    print(f"[BIOMEDGRAPHICA] Loading mapping from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=['BioMedGraphica_Conn_ID', 'HGNC_Symbol', 'Ensembl_Gene_ID', 'Gene_Name'])

    # Main mapping: HGNC_Symbol -> BioMedGraphica_Conn_ID
    hgnc_to_conn_id = {}
    # Alias mapping: various possible gene names -> HGNC_Symbol (for matching)
    gene_aliases = {}

    for _, row in df.iterrows():
        hgnc = str(row['HGNC_Symbol']).strip()
        conn_id = str(row['BioMedGraphica_Conn_ID']).strip()

        if pd.isna(row['HGNC_Symbol']) or hgnc == 'nan' or hgnc == '':
            continue

        hgnc_to_conn_id[hgnc] = conn_id

        # Add various possible aliases for matching
        gene_aliases[hgnc] = hgnc  # Self
        gene_aliases[hgnc.upper()] = hgnc  # Uppercase
        gene_aliases[hgnc.lower()] = hgnc  # Lowercase

        # Ensembl ID also serves as an alias
        if pd.notna(row['Ensembl_Gene_ID']):
            ensembl_ids = str(row['Ensembl_Gene_ID']).split(';')
            for eid in ensembl_ids:
                eid = eid.strip()
                if eid:
                    gene_aliases[eid] = hgnc

    print(f"[BIOMEDGRAPHICA] Loaded {len(hgnc_to_conn_id)} HGNC symbols")
    return hgnc_to_conn_id, gene_aliases


def map_genes_to_hgnc(gene_names: List[str],
                      gene_aliases: Dict[str, str],
                      hgnc_to_conn_id: Dict[str, str]) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Map gene name list to HGNC standard symbols.

    Args:
        gene_names: List of original gene names (from adata.var_names)
        gene_aliases: Alias -> HGNC_Symbol mapping
        hgnc_to_conn_id: HGNC_Symbol -> BioMedGraphica_Conn_ID mapping

    Returns:
        mapped_genes: List of successfully mapped HGNC_Symbols
        conn_ids: List of corresponding BioMedGraphica_Conn_IDs
        original_to_hgnc: {Original Gene Name: HGNC_Symbol} mapping (for later index conversion)
    """
    mapped_genes = []
    conn_ids = []
    original_to_hgnc = {}
    unmapped = []

    for gene in gene_names:
        gene_str = str(gene).strip()

        # Attempt direct match
        if gene_str in gene_aliases:
            hgnc = gene_aliases[gene_str]
            if hgnc in hgnc_to_conn_id:
                mapped_genes.append(hgnc)
                conn_ids.append(hgnc_to_conn_id[hgnc])
                original_to_hgnc[gene] = hgnc
                continue

        # Attempt uppercase match
        if gene_str.upper() in gene_aliases:
            hgnc = gene_aliases[gene_str.upper()]
            if hgnc in hgnc_to_conn_id:
                mapped_genes.append(hgnc)
                conn_ids.append(hgnc_to_conn_id[hgnc])
                original_to_hgnc[gene] = hgnc
                continue

        unmapped.append(gene_str)

    print(f"[GENE MAPPING] Mapped: {len(mapped_genes)}/{len(gene_names)} genes")
    if unmapped:
        print(f"[GENE MAPPING] Unmapped examples: {unmapped[:10]}...")

    return mapped_genes, conn_ids, original_to_hgnc


def build_gene_vocab(gene_names: List[str],
                     conn_ids: Optional[List[str]] = None) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Build gene vocabulary: gene identifier -> integer ID.

    When conn_ids are provided:
        - vocab keys use BioMedGraphica_Conn_ID (e.g., BMGC_GN00001)
        - Thus output tokens directly correspond to Knowledge Graph node IDs.

    When conn_ids are NOT provided:
        - vocab keys use original gene names.

    Special tokens:
        <pad>: 0 - used for padding short sequences to fixed length
        <cls>: 1 - used for classification tasks (like BERT CLS token)

    Ordinary gene IDs start from 2.

    Args:
        gene_names: List of gene names (HGNC_Symbol or original names)
        conn_ids: List of corresponding BioMedGraphica_Conn_IDs (optional)

    Returns:
        vocab: {BioMedGraphica_Conn_ID or Gene Name: gene_id} vocabulary
        conn_to_hgnc: {BioMedGraphica_Conn_ID: HGNC_Symbol} mapping (for explanation)
    """
    vocab = {"<pad>": 0, "<cls>": 1}
    conn_to_hgnc = {"<pad>": "<pad>", "<cls>": "<cls>"}

    for i, g in enumerate(gene_names):
        # If conn_ids exist, use Conn_ID as vocab key
        if conn_ids is not None and i < len(conn_ids):
            key = conn_ids[i]  # Use BioMedGraphica_Conn_ID
        else:
            key = g  # Use original gene name

        if key not in vocab:
            gid = len(vocab)
            vocab[key] = gid
            conn_to_hgnc[key] = g  # Record Conn_ID to HGNC mapping for explanation

    return vocab, conn_to_hgnc

def encode_labels(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode categorical labels as integers (LabelEncoder).

    Returns:
        y: Encoded integer array
        mapping: {label_name: integer_ID} mapping dictionary
    """
    vals = series.astype(str).fillna("NA").values
    uniq = pd.unique(vals).tolist()
    mapping = {k: i for i, k in enumerate(uniq)}
    y = np.array([mapping[v] for v in vals], dtype=np.int32)
    return y, mapping

def csr_row_to_topk_tokens(row: sp.csr_matrix,
                           gene_ids_array: np.ndarray,
                           edges: np.ndarray,
                           topk: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Core Tokenization Function: Convert single cell expression vector to token sequence.

    Process:
        1. Extract non-zero elements (expressed genes) from sparse matrix.
        2. Keep Top-K highest expressed genes.
        3. Use per-gene quantile edges to bin expression values.
        4. Sort by expression descending (high expression genes first).

    Args:
        row: 1×G CSR sparse matrix (single cell expression values)
        gene_ids_array: int32 [G], gene index -> vocab_id mapping
        edges: float32 [G, n_bins-1], quantile edges for each gene
        topk: Max number of genes to keep

    Returns:
        gene_ids: Gene ID array (sorted by expression descending)
        bin_ids: Corresponding bin values
        length: Effective sequence length

    Algorithm Optimization:
        - argpartition: O(n) complexity for Top-K, faster than full sort O(nlogn)
        - searchsorted: O(log(n_bins)) binary search for binning
    """
    # Extract non-zero elements: idx is gene index, val is expression value
    idx = row.indices
    val = row.data

    # Return immediately for empty cells
    if val.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int16), 0

    # If non-zero genes > topk, use argpartition for efficient Top-K
    if val.size > topk:
        sel = np.argpartition(val, -topk)[-topk:]  # Get indices of the largest topk elements
        idx = idx[sel]
        val = val[sel]

    # Per-gene binning: Each gene uses its own quantile edges
    bins = np.empty(val.shape[0], dtype=np.int16)
    for t, j in enumerate(idx):
        # searchsorted: Find position where val[t] should be inserted in edges[j]
        # side="right": Values equal to edge fall into the next bin
        bins[t] = np.searchsorted(edges[j], val[t], side="right")

    # Gene Index -> Vocab ID
    g = gene_ids_array[idx].astype(np.int32)

    # Sort by expression descending (high expression genes placed at start of sequence)
    order = np.argsort(-val)
    g = g[order]
    bins = bins[order]

    return g, bins, int(g.shape[0])


# =============================================================================
# Core Steps
# =============================================================================

def step_hvg(h5ad_path: Path,
             out_dir: Path,
             batch_key: str,
             n_top_genes: int,
             hvg_sample_total: int,
             seed: int, min_genes: int, max_genes: int,
             min_counts: int, max_counts: int,
             max_pct_mt: float, mt_prefix: str,
             qc_chunk_size: int) -> List[str]:
    """
    Step 1: Highly Variable Genes (HVG) selection

    Output: hvg_genes.txt (one gene name per line)
    """
    print("[HVG] Opening backed AnnData for obs sampling ...")
    adata_b = sc.read_h5ad(str(h5ad_path), backed="r")
    obs = adata_b.obs.copy()

    #keep_idx = filter_cell_indices(obs, filter_doublets=True)
    keep_idx = filter_cell_indices_qc_backed(
        adata_b,
        filter_doublets=True,
        doublet_col="predicted_doublet",
        min_genes=min_genes,
        max_genes=max_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        mt_prefix=mt_prefix,
        chunk_size=qc_chunk_size, 
    )
    sample_total = min(hvg_sample_total, keep_idx.size)
    print(f"[HVG] Sampling {sample_total} cells stratified by '{batch_key}' (from {keep_idx.size} kept cells).")

    obs_kept = obs.iloc[keep_idx]
    sample_rel = stratified_sample_indices(obs_kept, key=batch_key, n_total=sample_total, seed=seed)
    sample_idx = keep_idx[sample_rel]

    print("[HVG] Loading sampled cells into memory ...")
    adata_sub = adata_b[sample_idx, :].to_memory()
    if sp.issparse(adata_sub.X):
        adata_sub.X = adata_sub.X.tocsr()
    normalize_log1p_inplace(adata_sub)

    print("[HVG] Running scanpy highly_variable_genes (seurat_v3) ...")
    sc.pp.highly_variable_genes(
        adata_sub,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key=batch_key
    )
    hvg_mask = adata_sub.var["highly_variable"].values
    hvg_genes = adata_sub.var_names[hvg_mask].tolist()

    out_file = out_dir / "hvg_genes.txt"
    out_file.write_text("\n".join(hvg_genes))
    print(f"[HVG] Saved {len(hvg_genes)} genes -> {out_file}")

    adata_b.file.close()
    return hvg_genes

def step_bins_edges(h5ad_path: Path,
                    out_dir: Path,
                    hvg_genes: List[str],
                    edge_sample_total: int,
                    n_bins: int,
                    seed: int, min_genes: int, max_genes: int,
                    min_counts: int, max_counts: int,
                    max_pct_mt: float, mt_prefix: str,
                    qc_chunk_size: int) -> np.ndarray:
    """
    Step 3: Compute Per-gene quantile bin edges

    Binning principle (e.g., n_bins=32):
        - Compute 31 internal quantiles: 3.2%, 6.4%, ..., 96.8%.
        - Assign to bin based on which quantiles the expression value falls between.
        - bin_id range: 0 to n_bins-1.

    Output: gene_bin_edges.npy [G, n_bins-1]
    """
    print("[BINS] Opening backed AnnData for edge estimation ...")
    adata_b = sc.read_h5ad(str(h5ad_path), backed="r")
    obs = adata_b.obs.copy()
    keep_idx = filter_cell_indices_qc_backed(
        adata_b,
        filter_doublets=True,
        doublet_col="predicted_doublet",
        min_genes=min_genes,
        max_genes=max_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        mt_prefix=mt_prefix,
        chunk_size=qc_chunk_size,
    )

    sample_total = min(edge_sample_total, keep_idx.size)
    rng = set_seed(seed + 17)
    sample_idx = rng.choice(keep_idx, size=sample_total, replace=False)

    print(f"[BINS] Loading {sample_total} cells × {len(hvg_genes)} HVG genes into memory ...")
    adata_sub = adata_b[sample_idx, hvg_genes].to_memory()
    if sp.issparse(adata_sub.X):
        adata_sub.X = adata_sub.X.tocsr()
    normalize_log1p_inplace(adata_sub)

    X = adata_sub.X.tocsr() if sp.issparse(adata_sub.X) else np.asarray(adata_sub.X)

    qs = np.linspace(0, 1, n_bins + 1)[1:-1]  # internal quantiles
    G = adata_sub.n_vars
    edges = np.zeros((G, len(qs)), dtype=np.float32)

    print(f"[BINS] Computing per-gene quantile edges: G={G}, bins={n_bins} ...")
    for j in tqdm(range(G), desc="[BINS] quantiles", unit="gene"):
        col = X[:, j]
        if sp.issparse(col):
            col = col.toarray().reshape(-1)
        else:
            col = np.asarray(col).reshape(-1)
        edges[j] = np.quantile(col, qs)

    out_file = out_dir / "gene_bin_edges.npy"
    np.save(out_file, edges)
    print(f"[BINS] Saved edges {edges.shape} -> {out_file}")

    adata_b.file.close()
    return edges

def step_tokenize_shards(h5ad_path: Path,
                         out_dir: Path,
                         hvg_genes: List[str],
                         hvg_genes_for_vocab: List[str],
                         vocab: Dict[str, int],
                         edges: np.ndarray,
                         topk: int,
                         chunk_size: int,
                         target: Optional[str],
                         save_domain_ids: bool,
                         domain_key: str,
                         seed: int, min_genes: int, max_genes: int,
                         min_counts: int, max_counts: int,
                         max_pct_mt: float, mt_prefix: str,
                         qc_chunk_size: int,
                         text_cols: Optional[str] = None) -> None:
    """
    Step 4: Tokenize all cells and save as shards (most time-consuming step)

    Tokenization Process (per cell):
        1. Extract non-zero expressed genes.
        2. Keep Top-K highest expressed genes.
        3. Gene Name -> vocab ID.
        4. Expression value -> bin ID (using per-gene edges).
        5. Sort by expression descending.
        6. Pad to fixed length (topk).

    Sharding strategy:
        - Each shard contains chunk_size cells (default 50000).
        - Use npz compression to save disk space.
        - Facilitates parallel loading during training.

    Args:
        hvg_genes: List of original gene names (for indexing adata)
        hvg_genes_for_vocab: Gene names in vocab (HGNC symbols, used for finding gene_id)

    Output structure:
        shards/
        ├── shard_00000.npz  (gene_ids, bin_ids, lengths, [y], [domain_id])
        ├── shard_00001.npz
        └── ...
        label_map.json      (if --target is specified)
        domain_map.json     (if --save_domain_ids is enabled)
    """
    print("[TOK] Opening backed AnnData for tokenization ...")
    adata_b = sc.read_h5ad(str(h5ad_path), backed="r")
    obs = adata_b.obs.copy()

    keep_idx = filter_cell_indices_qc_backed(
        adata_b,
        filter_doublets=True,
        doublet_col="predicted_doublet",
        min_genes=min_genes,
        max_genes=max_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        mt_prefix=mt_prefix,
        chunk_size=qc_chunk_size,
    )
    print(f"[TOK] Kept cells: {keep_idx.size} / {len(obs)} (filtered predicted_doublet).")

    # label encoding (optional)
    # y_all = None
    # if target is not None:
    #     if target not in obs.columns:
    #         raise ValueError(f"--target '{target}' not found in obs.columns")
    #     y_all, label_map = encode_labels(obs[target].iloc[keep_idx])
    #     (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    #     print(f"[TOK] Saved label map -> {out_dir / 'label_map.json'} (classes={len(label_map)})")

    text_cols_list = None
    text_cols_missing = []  # Record missing columns
    if text_cols is not None and str(text_cols).strip() != "":
        text_cols_list = [c.strip() for c in text_cols.split(",") if c.strip()]

        # Check which columns exist and which are missing
        text_cols_missing = [c for c in text_cols_list if c not in obs.columns]
        text_cols_exist = [c for c in text_cols_list if c in obs.columns]

        if text_cols_missing:
            print(f"[TOK] Warning: missing columns will be filled with 'none': {text_cols_missing}")
        print(f"[TOK] Will save text_data columns: {text_cols_list}")
        print(f"[TOK]   - existing: {text_cols_exist}")
        print(f"[TOK]   - missing (fill 'none'): {text_cols_missing}")

    # domain ids (optional, downstream)
    domain_all = None
    if save_domain_ids:
        if domain_key not in obs.columns:
            raise ValueError(f"--domain_key '{domain_key}' not found in obs.columns")
        domain_all, domain_map = encode_labels(obs[domain_key].iloc[keep_idx])
        (out_dir / "domain_map.json").write_text(json.dumps(domain_map, indent=2))
        print(f"[TOK] Saved domain map -> {out_dir / 'domain_map.json'} (domains={len(domain_map)})")

    # Use hvg_genes_for_vocab (HGNC symbols) to look up vocab ID
    gene_ids_array = np.array([vocab[g] for g in hvg_genes_for_vocab], dtype=np.int32)

    n_cells = keep_idx.size
    n_shards = (n_cells + chunk_size - 1) // chunk_size
    print(f"[TOK] Writing shards: chunk_size={chunk_size}, shards={n_shards}")

    ensure_dir(out_dir / "shards")

    for shard in tqdm(range(n_shards), desc="[TOK] shards", unit="shard"):
        start = shard * chunk_size
        end = min((shard + 1) * chunk_size, n_cells)
        idx = keep_idx[start:end]

        X_chunk = adata_b[idx, hvg_genes].X
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.tocsr()
        else:
            X_chunk = sp.csr_matrix(X_chunk)

        N = X_chunk.shape[0]
        gene_mat = np.zeros((N, topk), dtype=np.int32)   # pad gene_id=0
        bin_mat  = np.zeros((N, topk), dtype=np.int16)   # pad bin_id=0
        lengths  = np.zeros((N,), dtype=np.int16)

        for i in range(N):
            row = X_chunk.getrow(i)
            g, b, L = csr_row_to_topk_tokens(row, gene_ids_array, edges, topk=topk)
            lengths[i] = L
            if L > 0:
                L2 = min(topk, L)
                gene_mat[i, :L2] = g[:L2]
                bin_mat[i, :L2]  = b[:L2]

        shard_path = out_dir / "shards" / f"shard_{shard:05d}.npz"
        payload = dict(gene_ids=gene_mat, bin_ids=bin_mat, lengths=lengths)

        if text_cols_list is not None:
            # Build meta_df: read existing columns from obs, fill missing ones with "none"
            meta_dict = {}
            for col in text_cols_list:
                if col in obs.columns:
                    meta_dict[col] = obs.loc[idx, col].values
                else:
                    # Fill missing column with "none"
                    meta_dict[col] = np.full(len(idx), "none", dtype=object)

            meta_df = pd.DataFrame(meta_dict, index=idx)
            # Convert all to string, fill missing values with "none" (avoid nan)
            meta_df = meta_df.astype("string").fillna("none")

            # Convert to numpy Unicode array, avoid object/pickle
            text_data = meta_df.to_numpy(dtype=str)
            text_data = np.asarray(text_data, dtype="U")  # Force unicode

            payload["text_data"] = text_data
            payload["text_cols"] = np.asarray(text_cols_list, dtype="U")

        # if target is not None:
        #     payload["y"] = y_all[start:end]

        if save_domain_ids:
            payload["domain_id"] = domain_all[start:end]

        np.savez_compressed(shard_path, **payload)

    adata_b.file.close()
    print(f"[TOK] Done. Shards are under: {out_dir / 'shards'}")


# =============================================================================
# Command Line Interface (CLI)
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Argument groups:
        Input/Output: --h5ad, --out_dir
        HVG related: --batch_key, --n_top_genes, --hvg_sample_total
        Binning related: --n_bins, --edge_sample_total
        Sequence related: --topk, --chunk_size
        Label related: --target, --save_domain_ids, --domain_key
    """
    p = argparse.ArgumentParser(description="Huge AnnData -> gene-token shards (server-friendly).")

    # === Input/Output ===
    p.add_argument("--h5ad", type=str, required=True,
                   help="Input h5ad file path")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory")

    # === HVG Selection ===
    p.add_argument("--batch_key", type=str, default="datasetID",
                   help="Batch column name for batch-aware HVG (default: datasetID)")
    p.add_argument("--n_top_genes", type=int, default=3000,
                   help="Number of HVG genes to select (default: 3000)")
    p.add_argument("--hvg_sample_total", type=int, default=200000,
                   help="Total cells to sample for HVG calculation (default: 200000)")
    p.add_argument("--edge_sample_total", type=int, default=200000,
                   help="Total cells to sample for bin edge estimation (default: 200000)")

    # === Binning ===
    p.add_argument("--n_bins", type=int, default=32,
                   help="Number of bins for expression values (default: 32)")

    # === Sequence ===
    p.add_argument("--topk", type=int, default=1024,
                   help="Number of Top-K genes per cell = sequence length (default: 1024)")
    p.add_argument("--chunk_size", type=int, default=50000,
                   help="Number of cells per shard (default: 50000)")

    # === Labels ===
    p.add_argument("--target", type=str, default=None,
                    help="Prediction target column (e.g., compartment, cellType1). If not set, labels are not saved.")
    p.add_argument("--text_cols", type=str, default="tissue,region,sampleType,compartment,cellType1,cellType2,majorCluster,subCluster,sex,age,perturbType,disease,drug,dose",
        help="Comma-separated obs column names to save as text_data, e.g.: tissue,region,cellType1,cellType2")
    p.add_argument("--save_domain_ids", action="store_true",
                   help="Save batch IDs for downstream use (default not saved for pretrain)")
    p.add_argument("--domain_key", type=str, default="datasetID",
                   help="Source column for batch IDs (default: datasetID)")

    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")

    # === BioMedGraphica Gene Mapping ===
    p.add_argument("--biomedgraphica", type=str, default=None,
                   help="Path to BioMedGraphica_Conn_Gene.csv, used to map gene names to HGNC standard symbols")
    
        # === QC Parameters ===
    p.add_argument("--min_genes", type=int, default=500)
    p.add_argument("--max_genes", type=int, default=8000)
    p.add_argument("--min_counts", type=int, default=1000)
    p.add_argument("--max_counts", type=int, default=100000)
    p.add_argument("--max_pct_mt", type=float, default=20.0)
    p.add_argument("--mt_prefix", type=str, default="MT-",
                   help="Mitochondrial gene prefix (commonly MT- for human; mt- for mouse)")
    p.add_argument("--qc_chunk_size", type=int, default=50000,
                   help="Chunk size when scanning X for QC (default: 50000)")

    return p.parse_args()

def main():
    """
    Main function: Execute 4 processing steps sequentially.

    Process flow:
        Step 0 (Optional): Load BioMedGraphica mapping.
        Step 1: HVG selection -> hvg_genes.txt
        Step 1.5 (Optional): Map gene names to HGNC standard symbols.
        Step 2: Build vocabulary -> gene_vocab.json + id_to_biomedgraphica.json
        Step 3: Compute bin edges -> gene_bin_edges.npy
        Step 4: Tokenize and shard -> shards/*.npz
    """
    args = parse_args()
    h5ad_path = Path(args.h5ad)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # =========================================================================
    # Step 0 Load BioMedGraphica Gene Mapping
    # =========================================================================
    hgnc_to_conn_id = None
    gene_aliases = None
    if args.biomedgraphica:
        biomedgraphica_path = Path(args.biomedgraphica)
        if not biomedgraphica_path.exists():
            raise FileNotFoundError(f"BioMedGraphica file not found: {biomedgraphica_path}")
        hgnc_to_conn_id, gene_aliases = load_biomedgraphica_mapping(biomedgraphica_path)

    # =========================================================================
    # Step 1: HVG Selection
    # Input: Original h5ad -> Output: hvg_genes.txt
    # =========================================================================
    hvg_genes_original = step_hvg(
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        batch_key=args.batch_key,
        n_top_genes=args.n_top_genes,
        hvg_sample_total=args.hvg_sample_total,
        seed=args.seed,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        max_pct_mt=args.max_pct_mt,
        mt_prefix=args.mt_prefix,
        qc_chunk_size=args.qc_chunk_size,
    )

    # =========================================================================
    # Step 2 Map gene names to BioMedGraphica_Conn_ID
    # =========================================================================
    # hvg_genes_for_adata: Original gene names for indexing adata
    # hvg_genes_for_vocab: Identifiers for vocab building (BioMedGraphica_Conn_ID or original names)
    hvg_genes_for_adata = hvg_genes_original
    hvg_genes_for_vocab = hvg_genes_original  # Default to original gene names

    if hgnc_to_conn_id is not None:
        print("\n" + "="*60)
        print("[GENE MAPPING] Mapping gene names to HGNC symbols...")
        print("="*60)

        # Map gene names
        mapped_hgnc = []
        mapped_conn_ids = []
        mapped_original = []  # Corresponding original gene names (for indexing adata)

        unmapped = []
        for gene in hvg_genes_original:
            gene_str = str(gene).strip()

            # Attempt match
            hgnc = None
            if gene_str in gene_aliases:
                hgnc = gene_aliases[gene_str]
            elif gene_str.upper() in gene_aliases:
                hgnc = gene_aliases[gene_str.upper()]

            if hgnc and hgnc in hgnc_to_conn_id:
                mapped_hgnc.append(hgnc)
                mapped_conn_ids.append(hgnc_to_conn_id[hgnc])
                mapped_original.append(gene)  # Save original gene name
            else:
                unmapped.append(gene_str)

        print(f"[GENE MAPPING] Mapped: {len(mapped_hgnc)}/{len(hvg_genes_original)} genes")
        if unmapped:
            print(f"[GENE MAPPING] Unmapped examples: {unmapped[:10]}...")

        # Keep only mapped genes
        hvg_genes_for_adata = mapped_original      # Original gene names for indexing adata
        hvg_genes_for_vocab = mapped_conn_ids      # BioMedGraphica_Conn_ID for vocab (as KG node IDs)
        hvg_genes_hgnc = mapped_hgnc               # HGNC symbols for saving mapping relations

        # Save mapping relations
        mapping_info = {
            'original_to_hgnc': dict(zip(mapped_original, mapped_hgnc)),
            'hgnc_to_conn_id': dict(zip(mapped_hgnc, mapped_conn_ids)),
            'conn_id_to_hgnc': dict(zip(mapped_conn_ids, mapped_hgnc)),  # Reverse mapping for explanation
            'total_original': len(hvg_genes_original),
            'total_mapped': len(mapped_hgnc),
            'mapping_rate': f"{len(mapped_hgnc)/len(hvg_genes_original)*100:.1f}%"
        }
        (out_dir / "gene_mapping_info.json").write_text(json.dumps(mapping_info, indent=2, ensure_ascii=False))
        print(f"[GENE MAPPING] Saved mapping info -> {out_dir / 'gene_mapping_info.json'}")

        # Save HGNC gene list and Conn_ID list
        (out_dir / "hvg_genes_hgnc.txt").write_text("\n".join(hvg_genes_hgnc))
        print(f"[GENE MAPPING] Saved HGNC gene list ({len(hvg_genes_hgnc)} genes) -> {out_dir / 'hvg_genes_hgnc.txt'}")

        (out_dir / "hvg_genes_conn_id.txt").write_text("\n".join(hvg_genes_for_vocab))
        print(f"[GENE MAPPING] Saved BioMedGraphica_Conn_ID list ({len(hvg_genes_for_vocab)} genes) -> {out_dir / 'hvg_genes_conn_id.txt'}")

    # =========================================================================
    # Step 3: Build Gene Vocabulary
    # When using BioMedGraphica mapping: vocab key is directly BioMedGraphica_Conn_ID
    # Thus the output gene_ids directly correspond to Knowledge Graph node IDs.
    # =========================================================================
    vocab, conn_to_hgnc = build_gene_vocab(hvg_genes_for_vocab, None)  # hvg_genes_for_vocab is already Conn_ID
    (out_dir / "gene_vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"[VOCAB] Saved vocab (size={len(vocab)}) -> {out_dir / 'gene_vocab.json'}")

    # Save Conn_ID -> HGNC_Symbol mapping (for interpreting model output)
    (out_dir / "conn_id_to_hgnc.json").write_text(json.dumps(conn_to_hgnc, indent=2))
    print(f"[VOCAB] Saved conn_id_to_hgnc mapping -> {out_dir / 'conn_id_to_hgnc.json'}")

    # =========================================================================
    # Step 4: Compute Per-gene Bin Edges
    # Input: Original h5ad + hvg_genes_for_adata (original gene names) -> Output: gene_bin_edges.npy
    # =========================================================================
    edges = step_bins_edges(
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        hvg_genes=hvg_genes_for_adata,  # Use original gene names to index adata
        edge_sample_total=args.edge_sample_total,
        n_bins=args.n_bins,
        seed=args.seed,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        max_pct_mt=args.max_pct_mt,
        mt_prefix=args.mt_prefix,
        qc_chunk_size=args.qc_chunk_size,
    )

    # =========================================================================
    # Step 5: Tokenize All Cells and Shard Save
    # Input: Original h5ad + vocab + edges -> Output: shards/*.npz
    # Note: hvg_genes_for_adata used for indexing adata, hvg_genes_for_vocab used for vocab lookup
    # =========================================================================
    step_tokenize_shards(
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        hvg_genes=hvg_genes_for_adata,           # Original gene names, used for indexing adata
        hvg_genes_for_vocab=hvg_genes_for_vocab, # HGNC symbols, used for looking up vocab ID
        vocab=vocab,
        edges=edges,
        topk=args.topk,
        chunk_size=args.chunk_size,
        target=args.target,
        save_domain_ids=args.save_domain_ids,
        domain_key=args.domain_key,
        seed=args.seed,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        max_pct_mt=args.max_pct_mt,
        mt_prefix=args.mt_prefix,
        qc_chunk_size=args.qc_chunk_size,
        text_cols=args.text_cols,
    )

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "="*60)
    print("All done!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {out_dir / 'hvg_genes.txt'}              <- HVG list (Original gene names)")
    print(f"  - {out_dir / 'gene_bin_edges.npy'}         <- Per-gene bin edges")
    print(f"  - {out_dir / 'shards/'}                    <- Tokenized data shards")
    if args.biomedgraphica:
        print(f"  - {out_dir / 'gene_vocab.json'}            <- Gene Vocabulary (BioMedGraphica_Conn_ID -> ID)")
        print(f"  - {out_dir / 'hvg_genes_conn_id.txt'}      <- BioMedGraphica_Conn_ID list")
        print(f"  - {out_dir / 'hvg_genes_hgnc.txt'}         <- HGNC standard symbol list")
        print(f"  - {out_dir / 'conn_id_to_hgnc.json'}       <- Conn_ID -> HGNC mapping (for explanation)")
        print(f"  - {out_dir / 'gene_mapping_info.json'}     <- Gene mapping details")
    else:
        print(f"  - {out_dir / 'gene_vocab.json'}            <- Gene Vocabulary (Original gene names -> ID)")
        print(f"  - {out_dir / 'conn_id_to_hgnc.json'}       <- Gene name mapping")

if __name__ == "__main__":
    main()

"""
python data_process_server.py \
  --h5ad igt_s9_fine_counts.h5ad \
  --out_dir out_pretrain_qc \
  --batch_key datasetID \
  --target compartment \
  --n_top_genes 4000 \
  --n_bins 32 \
  --topk 1024 \
  --chunk_size 50000 \
  --min_genes 500 --max_genes 8000 \
  --min_counts 1000 --max_counts 100000 \
  --max_pct_mt 20 \
  --mt_prefix MT-
"""