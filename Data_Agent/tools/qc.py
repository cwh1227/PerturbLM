"""
Uniform QC and HVG selection tools.

All tools read from / write to the standardized parquet files produced
by read_* tools.  No format-specific logic except where expression data
must be accessed (compute_qc_metrics, select_hvg).

compute_qc_metrics  (call when n_genes / n_counts / pct_mt are NaN)
──────────────────
Reads  intermediate/cell_meta.parquet  +  expression data (via data_source.json)
Writes intermediate/cell_meta.parquet  (fills n_genes, n_counts, pct_mt)

qc_filter_cells
───────────────
Reads  intermediate/cell_meta.parquet
Writes intermediate/cell_meta.parquet  (filtered in place)

All datasets use the same column names because read_* already standardized them:
  qc_pass   bool    — primary QC flag   (always present)
  n_genes   int     — detected genes    (NaN if not pre-computed → call compute_qc_metrics)
  n_counts  int     — total counts      (NaN if not pre-computed → call compute_qc_metrics)
  pct_mt    float   — % mitochondrial   (NaN if not pre-computed → call compute_qc_metrics)

select_hvg
──────────
Reads  intermediate/gene_meta.parquet  +  expression data (via data_source.json)
Writes intermediate/gene_meta.parquet  (adds/updates is_hvg column)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from Agent.state import cell_meta_path, data_source_path, gene_meta_path


# ---------------------------------------------------------------------------
# 1. Compute QC metrics from expression matrix (when not pre-computed)
# ---------------------------------------------------------------------------

@tool
def compute_qc_metrics(
    mt_prefix:        str = "MT-",
    intermediate_dir: str = "",
) -> str:
    """
    Compute per-cell QC metrics (n_genes, n_counts, pct_mt) from the expression
    matrix and write them into cell_meta.parquet.

    Call this when inspect / read output shows that n_genes / n_counts / pct_mt
    are NaN (i.e., not pre-computed in the original file).
    Not needed for bulk datasets (L1000 gctx) — those metrics are not meaningful.

    mt_prefix — prefix used to identify mitochondrial genes in gene_symbol
                (default "MT-" for human; use "mt-" for mouse).

    Reads  : cell_meta.parquet, gene_meta.parquet, data_source.json → expression data
    Writes : cell_meta.parquet  (updates n_genes, n_counts, pct_mt columns)
    """
    try:
        cell_meta = pd.read_parquet(cell_meta_path(intermediate_dir))
        gene_meta = pd.read_parquet(gene_meta_path(intermediate_dir))
        data_src  = json.loads(data_source_path(intermediate_dir).read_text())
        fmt       = data_src.get("format", "")

        mt_mask = gene_meta["gene_symbol"].str.startswith(mt_prefix).values  # shape (n_genes,)

        n_cells  = len(cell_meta)
        n_genes_arr  = np.zeros(n_cells, dtype=np.int32)
        n_counts_arr = np.zeros(n_cells, dtype=np.float64)
        pct_mt_arr   = np.zeros(n_cells, dtype=np.float64)

        # ── h5ad ──────────────────────────────────────────────────────────
        if fmt == "h5ad":
            import anndata as ad
            import scipy.sparse as sp

            adata       = ad.read_h5ad(data_src["path"], backed="r")
            sid_to_row  = {sid: i for i, sid in enumerate(cell_meta.index.astype(str))}
            for i in range(0, len(adata), 2000):
                batch    = adata[i: i + 2000]
                sids     = batch.obs_names.astype(str).tolist()
                X        = batch.X
                if sp.issparse(X):
                    X = X.toarray()
                X = X.astype(np.float64)
                for local_j, sid in enumerate(sids):
                    row = sid_to_row.get(sid)
                    if row is None:
                        continue
                    counts            = X[local_j]
                    total             = counts.sum()
                    n_genes_arr[row]  = int((counts > 0).sum())
                    n_counts_arr[row] = total
                    pct_mt_arr[row]   = float(counts[mt_mask].sum() / total * 100) if total > 0 else 0.0
            adata.file.close()

        # ── gctx ──────────────────────────────────────────────────────────
        elif fmt == "gctx":
            import h5py

            with h5py.File(data_src["path"], "r") as f:
                mat        = f["/0/DATA/0/matrix"]
                sample_ids = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in f["/0/META/COL/id"][:]
                ]
                sid_to_row = {sid: i for i, sid in enumerate(cell_meta.index.astype(str))}
                for i in range(0, len(sample_ids), 5000):
                    batch_ids = sample_ids[i: i + 5000]
                    X = mat[i: i + 5000, :].astype(np.float64)
                    for local_j, sid in enumerate(batch_ids):
                        row = sid_to_row.get(sid)
                        if row is None:
                            continue
                        counts            = X[local_j]
                        total             = counts.sum()
                        n_genes_arr[row]  = int((counts > 0).sum())
                        n_counts_arr[row] = total
                        pct_mt_arr[row]   = float(counts[mt_mask].sum() / total * 100) if total > 0 else 0.0

        # ── HuggingFace ───────────────────────────────────────────────────
        elif fmt == "hf":
            from datasets import load_from_disk

            split   = data_src.get("subset_split", "train")
            expr_ds = load_from_disk(str(Path(data_src["path"]) / "expression_data"))
            expr_ds = expr_ds[split] if hasattr(expr_ds, "keys") else expr_ds

            sid_to_row  = {sid: i for i, sid in enumerate(cell_meta.index.astype(str))}
            tok_to_idx  = {int(t): i for i, t in enumerate(gene_meta["orig_token_id"])} \
                          if "orig_token_id" in gene_meta.columns else \
                          {int(t): i for i, t in enumerate(gene_meta["orig_id"])}
            id_col = next(
                (c for c in ("BARCODE_SUB_LIB_ID", "barcode", "cell_id", "sample_id")
                 if c in expr_ds.column_names), None
            )
            tok_col = next((c for c in ("gene_token_ids", "token_ids")
                            if c in expr_ds.column_names), None)
            val_col = next((c for c in ("values", "counts")
                            if c in expr_ds.column_names), None)

            if tok_col and val_col:
                for record in expr_ds:
                    sid = str(record[id_col]) if id_col else None
                    row = sid_to_row.get(sid) if sid else None
                    if row is None:
                        continue
                    total = mt_total = 0.0
                    n_g   = 0
                    for tid, val in zip(record[tok_col], record[val_col]):
                        idx = tok_to_idx.get(int(tid))
                        if idx is not None:
                            v      = float(val)
                            total += v
                            n_g   += 1
                            if mt_mask[idx]:
                                mt_total += v
                    n_genes_arr[row]  = n_g
                    n_counts_arr[row] = total
                    pct_mt_arr[row]   = float(mt_total / total * 100) if total > 0 else 0.0

        else:
            return json.dumps({"error": f"Unknown format '{fmt}' in data_source.json"})

        cell_meta["n_genes"]  = n_genes_arr
        cell_meta["n_counts"] = n_counts_arr
        cell_meta["pct_mt"]   = pct_mt_arr
        cell_meta.to_parquet(cell_meta_path(intermediate_dir), index=True)

        return json.dumps({
            "status":        "ok",
            "n_cells":       n_cells,
            "median_genes":  float(np.median(n_genes_arr)),
            "median_counts": float(np.median(n_counts_arr)),
            "median_pct_mt": float(np.median(pct_mt_arr)),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 2. Cell / sample quality control  (completely format-agnostic)
# ---------------------------------------------------------------------------

@tool
def qc_filter_cells(
    min_genes:  int   = 0,
    max_genes:  int   = 0,
    min_counts: int   = 0,
    max_counts: int   = 0,
    max_pct_mt: float = 0.0,
    intermediate_dir: str = "",
) -> str:
    """
    Filter cells in cell_meta.parquet to retain only quality cells.

    Always removes cells where qc_pass=False (set by read_* from the
    dataset's own QC column, e.g., qc_pass=='1' for L1000, pass_filter=='full'
    for Tahoe, or True-for-all for h5ad).

    Additional numeric thresholds (applied only when > 0):
      min_genes  / max_genes   — filter on standardized n_genes column
      min_counts / max_counts  — filter on standardized n_counts column
      max_pct_mt               — filter on standardized pct_mt column

    Reads and overwrites intermediate/cell_meta.parquet.
    Returns JSON with before / after cell counts.
    """
    try:
        df = pd.read_parquet(cell_meta_path(intermediate_dir))
        n_before = len(df)

        mask = df["qc_pass"].astype(bool)

        def _apply(col, lo=None, hi=None):
            nonlocal mask
            if col not in df.columns:
                return
            s = pd.to_numeric(df[col], errors="coerce")
            if lo is not None and lo > 0:
                mask &= s.fillna(0) >= lo
            if hi is not None and hi > 0:
                mask &= s.fillna(0) <= hi

        _apply("n_genes",  lo=min_genes,  hi=max_genes)
        _apply("n_counts", lo=min_counts, hi=max_counts)
        if max_pct_mt > 0 and "pct_mt" in df.columns:
            mask &= pd.to_numeric(df["pct_mt"], errors="coerce").fillna(100.0) <= max_pct_mt

        df[mask].to_parquet(cell_meta_path(intermediate_dir), index=True)
        n_after = int(mask.sum())

        return json.dumps({
            "status":    "ok",
            "n_before":  n_before,
            "n_after":   n_after,
            "n_removed": n_before - n_after,
            "pct_kept":  f"{n_after / max(n_before, 1) * 100:.1f}%",
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 3. Highly variable gene selection
# ---------------------------------------------------------------------------

@tool
def select_hvg(
    n_top_genes: int = 3000,
    sample_size: int = 50000,
    intermediate_dir: str = "",
) -> str:
    """
    Select highly variable genes by computing per-gene expression variance.

    Reads gene_meta.parquet to get the gene list, reads expression data
    (format detected from data_source.json) to compute variance, then
    marks the top n_top_genes genes as is_hvg=True and writes gene_meta.parquet.

    Expression format dispatch (internal only — not exposed to the agent):
      h5ad  — samples adata.X in chunks
      gctx  — samples the HDF5 matrix
      hf    — samples HuggingFace expression_data sparse rows

    If n_top_genes <= 0, all genes are marked is_hvg=True (no filtering).
    For bulk L1000 (fixed 978 landmark genes) this step is optional.

    Returns JSON with HVG count and sampling info.
    """
    try:
        cell_meta = pd.read_parquet(cell_meta_path(intermediate_dir))
        gene_meta = pd.read_parquet(gene_meta_path(intermediate_dir))
        data_src  = json.loads(data_source_path(intermediate_dir).read_text())
        n_genes   = len(gene_meta)
        fmt       = data_src.get("format", "")
        kept_ids  = set(cell_meta.index.astype(str))

        if n_top_genes <= 0:
            gene_meta["is_hvg"]   = True
            gene_meta["variance"] = 0.0
            gene_meta.to_parquet(gene_meta_path(intermediate_dir), index=False)
            return json.dumps({"status": "ok", "n_hvg": n_genes,
                               "note": "all genes retained (n_top_genes=0)"})

        gene_sum  = np.zeros(n_genes, dtype=np.float64)
        gene_sum2 = np.zeros(n_genes, dtype=np.float64)
        gene_cnt  = np.zeros(n_genes, dtype=np.int64)
        sampled   = 0

        # ── h5ad ──────────────────────────────────────────────────────────
        if fmt == "h5ad":
            import anndata as ad
            import scipy.sparse as sp

            adata   = ad.read_h5ad(data_src["path"], backed="r")
            kept_indices = [i for i, sid in enumerate(adata.obs_names.astype(str)) if sid in kept_ids]
            rng = np.random.default_rng(42)
            indices = rng.choice(kept_indices, size=min(sample_size, len(kept_indices)), replace=False).tolist()
            for i in range(0, len(indices), 2000):
                batch = indices[i: i + 2000]
                X = adata[batch].X
                if sp.issparse(X):
                    X = X.toarray()
                X = X.astype(np.float64)
                gene_sum  += X.sum(axis=0)
                gene_sum2 += (X ** 2).sum(axis=0)
                gene_cnt  += X.shape[0]
            adata.file.close()
            sampled = len(indices)

        # ── gctx ──────────────────────────────────────────────────────────
        elif fmt == "gctx":
            import h5py

            with h5py.File(data_src["path"], "r") as f:
                mat   = f["/0/DATA/0/matrix"]
                sample_ids = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in f["/0/META/COL/id"][:]
                ]
                kept_indices = [i for i, sid in enumerate(sample_ids) if sid in kept_ids]
                rng = np.random.default_rng(42)
                indices = sorted(rng.choice(kept_indices, size=min(sample_size, len(kept_indices)), replace=False).tolist())
                for i in range(0, len(indices), 5000):
                    batch = indices[i: i + 5000]
                    X = mat[batch, :].astype(np.float64)
                    gene_sum  += X.sum(axis=0)
                    gene_sum2 += (X ** 2).sum(axis=0)
                    gene_cnt  += X.shape[0]
            sampled = len(indices)

        # ── HuggingFace ───────────────────────────────────────────────────
        elif fmt == "hf":
            from datasets import load_from_disk

            split    = data_src.get("subset_split", "train")
            expr_ds  = load_from_disk(
                str(Path(data_src["path"]) / "expression_data")
            )
            expr_ds = expr_ds[split] if hasattr(expr_ds, "keys") else expr_ds

            expr_id_col = next(
                (c for c in ("BARCODE_SUB_LIB_ID", "barcode", "cell_id", "sample_id")
                 if c in expr_ds.column_names),
                None,
            )
            if expr_id_col:
                candidate_indices = [
                    i for i, sid in enumerate(expr_ds[expr_id_col])
                    if str(sid) in kept_ids
                ]
            else:
                candidate_indices = list(range(expr_ds.num_rows))

            rng     = np.random.default_rng(42)
            indices = rng.choice(candidate_indices, size=min(sample_size, len(candidate_indices)), replace=False).tolist()
            sample_ds = expr_ds.select(indices)
            sampled  = len(indices)

            # orig_token_id → gene_meta row index
            if "orig_token_id" in gene_meta.columns:
                tok_to_idx = {int(t): i for i, t in enumerate(gene_meta["orig_token_id"])}
            else:
                tok_to_idx = {int(t): i for i, t in enumerate(gene_meta["orig_id"])}

            tok_col = next((c for c in ("gene_token_ids", "token_ids")
                            if c in sample_ds.column_names), None)
            val_col = next((c for c in ("values", "counts")
                            if c in sample_ds.column_names), None)
            if tok_col and val_col:
                for row in sample_ds:
                    for tid, val in zip(row[tok_col], row[val_col]):
                        idx = tok_to_idx.get(int(tid))
                        if idx is not None:
                            v = float(val)
                            gene_sum[idx]  += v
                            gene_sum2[idx] += v * v
                    gene_cnt += 1
        else:
            return json.dumps({"error": f"Unknown format '{fmt}' in data_source.json"})

        # Variance per gene
        with np.errstate(divide="ignore", invalid="ignore"):
            mean  = np.where(gene_cnt > 0, gene_sum  / gene_cnt, 0.0)
            mean2 = np.where(gene_cnt > 0, gene_sum2 / gene_cnt, 0.0)
            var   = np.maximum(mean2 - mean ** 2, 0.0)

        top_n   = min(n_top_genes, n_genes)
        top_set = set(np.argsort(var)[::-1][:top_n].tolist())

        gene_meta["variance"] = var
        gene_meta["is_hvg"]   = [i in top_set for i in range(n_genes)]
        gene_meta.to_parquet(gene_meta_path(intermediate_dir), index=False)

        return json.dumps({
            "status":        "ok",
            "n_hvg":         top_n,
            "n_total_genes": n_genes,
            "sampled_cells": sampled,
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
