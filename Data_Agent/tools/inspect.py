"""
File inspection tools.

These tools let the LLM understand the structure of any input file
(rows, columns, identifiers, metadata) before deciding which
pipeline steps to call.
"""

from __future__ import annotations

import json
import traceback

from langchain_core.tools import tool


def _resolve_dataset_split(ds, preferred_split: str | None = None):
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        split_name = preferred_split if preferred_split in splits else splits[0]
        return ds[split_name], splits
    return ds, ["(single)"]


@tool
def inspect_h5ad(path: str) -> str:
    """
    Inspect an AnnData .h5ad file without loading all data into memory.

    Returns a JSON summary:
      shape        - (n_cells, n_genes)
      obs_columns  - cell-level metadata column names
      var_columns  - gene-level metadata column names
      var_names    - first 20 gene identifiers (var_names)
      obs_sample   - first row of obs as {col: value} dict
      is_sparse    - whether X is stored as a sparse matrix
    """
    try:
        import anndata as ad
        import scipy.sparse as sp

        adata = ad.read_h5ad(path, backed="r")
        info = {
            "shape":       list(adata.shape),
            "obs_columns": adata.obs.columns.tolist(),
            "var_columns": adata.var.columns.tolist(),
            "var_names":   adata.var_names[:20].tolist(),
            "obs_sample":  adata.obs.iloc[0].to_dict() if len(adata.obs) > 0 else {},
            "is_sparse":   bool(sp.issparse(adata.X)),
        }
        adata.file.close()
        return json.dumps(info, default=str, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@tool
def inspect_gctx(path: str) -> str:
    """
    Inspect a CLUE/L1000 .gctx file (HDF5 format).

    Returns a JSON summary:
      matrix_shape  - (n_signatures, n_genes)
      entrez_ids    - first 20 Entrez IDs from the gene (ROW) axis
      sample_ids    - first 5 signature IDs from the column (COL) axis
    """
    try:
        import h5py

        with h5py.File(path, "r") as f:
            X   = f["/0/DATA/0/matrix"]
            rid = f["/0/META/ROW/id"]
            cid = f["/0/META/COL/id"]
            info = {
                "matrix_shape": list(X.shape),
                "entrez_ids":   [x.decode() for x in rid[:20]],
                "sample_ids":   [x.decode() for x in cid[:5]],
            }
        return json.dumps(info, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@tool
def inspect_hf_dataset(path: str, subset: str = "obs_metadata") -> str:
    """
    Inspect a HuggingFace Arrow dataset stored on disk (e.g., Tahoe-100M).

    Args:
        path   - Root directory of the HF dataset
        subset - Sub-folder to inspect.
                 Common values: 'obs_metadata', 'gene_metadata',
                 'expression_data', 'cell_line_metadata'

    Returns a JSON summary:
      num_rows     - total row count in the first split
      column_names - column names
      splits       - available split names
      sample       - first row as {col: value} dict (first 10 columns)
    """
    try:
        from datasets import load_from_disk
        from pathlib import Path

        ds = load_from_disk(str(Path(path) / subset))
        split_ds, splits = _resolve_dataset_split(ds)
        info = {
            "subset":       subset,
            "splits":       splits,
            "num_rows":     split_ds.num_rows,
            "column_names": split_ds.column_names,
            "sample":       {c: str(split_ds[c][0]) for c in split_ds.column_names[:10]},
        }
        return json.dumps(info, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@tool
def inspect_tsv(path: str, nrows: int = 5) -> str:
    """
    Inspect a TSV or CSV file.

    Returns a JSON summary:
      columns - column names
      rows    - first `nrows` rows as a list of dicts
    Auto-detects separator (tab for .txt / .tsv, comma otherwise).
    """
    try:
        import pandas as pd

        sep = "\t" if path.endswith((".txt", ".tsv")) else ","
        df  = pd.read_csv(path, sep=sep, nrows=nrows, dtype=str)
        return json.dumps(
            {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records")},
            default=str, indent=2,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
