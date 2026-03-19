"""
ATAC inspect tools — understand source format before reading.

Tools:
  inspect_atac_table  — flat peak-to-gene table (.csv/.tsv/.txt/.parquet)
  inspect_atac_bundle — scATAC bundle directory (peak_bc/gene_scores/ChromVar/...)
"""

from __future__ import annotations

import gzip
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from Agent.tools.atac_utils import (
    read_table,
    guess_column,
    _GENE_CANDIDATES,
    _DISTANCE_CANDIDATES,
    _OPENNESS_CANDIDATES,
)


def _read_mtx_shape_gz(mtx_gz: Path) -> tuple[int, int, int | None]:
    with gzip.open(mtx_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) == 3:
                r, c, nnz = map(int, parts)
                return r, c, nnz
            if len(parts) == 2:
                r, c = map(int, parts)
                return r, c, None
            raise ValueError(f"Unexpected MatrixMarket size line: {line}")
    raise ValueError(f"No size line found in {mtx_gz}")


def _bundle_schema_entry(folder: Path) -> dict:
    entry: dict = {"folder": str(folder)}

    obs = folder / "obs.csv"
    var = folder / "var.csv"
    mtx = folder / "counts.mtx.gz"

    if obs.exists():
        obs_df = pd.read_csv(obs, nrows=5)
        entry["obs_columns"] = [str(c) for c in obs_df.columns]
        entry["obs_n_cols"] = int(obs_df.shape[1])
        entry["obs_n_rows"] = int(sum(1 for _ in open(obs, "r", encoding="utf-8")) - 1)

    if var.exists():
        var_df = pd.read_csv(var, nrows=5)
        entry["var_columns"] = [str(c) for c in var_df.columns]
        entry["var_n_cols"] = int(var_df.shape[1])
        entry["var_n_rows"] = int(sum(1 for _ in open(var, "r", encoding="utf-8")) - 1)

    if mtx.exists():
        r, c, nnz = _read_mtx_shape_gz(mtx)
        entry["mtx_shape"] = [r, c]
        entry["mtx_nnz"] = None if nnz is None else int(nnz)

    return entry


@tool
def inspect_atac_table(path: str, nrows: int = 5) -> str:
    """
    Inspect an ATAC peak-to-gene flat table and suggest key columns.

    Returns:
      - n_rows, n_cols, columns
      - sample_rows
      - guessed_gene_col, guessed_distance_col, guessed_openness_col

    Args:
      path:  Input file path (.csv/.tsv/.txt/.parquet)
      nrows: Number of sample rows to return (default 5)
    """
    try:
        df = read_table(path)
        sample = df.head(max(1, nrows)).replace({np.nan: None}).to_dict(orient="records")
        return json.dumps(
            {
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "columns": [str(c) for c in df.columns],
                "guessed_gene_col": guess_column(df, _GENE_CANDIDATES),
                "guessed_distance_col": guess_column(df, _DISTANCE_CANDIDATES),
                "guessed_openness_col": guess_column(df, _OPENNESS_CANDIDATES),
                "sample_rows": sample,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@tool
def inspect_atac_bundle(dataset_dir: str) -> str:
    """
    Inspect a scATAC bundle directory.

    Expected subfolders: peak_bc, gene_scores, ChromVar, LSI_embedding, markerpeak_target.
    Each subfolder is inspected for counts.mtx.gz shape, obs.csv columns/rows,
    and var.csv columns/rows.

    Also returns recommended gene_col and distance_col for read_atac_bundle.

    Args:
      dataset_dir: Path to the bundle directory
    """
    try:
        base = Path(dataset_dir)
        if not base.exists():
            return json.dumps({"error": f"dataset_dir not found: {dataset_dir}"}, ensure_ascii=False)

        names = ["peak_bc", "gene_scores", "ChromVar", "LSI_embedding", "markerpeak_target"]
        info: dict = {
            "dataset_dir": str(base),
            "subfolders": {},
        }

        for name in names:
            folder = base / name
            if folder.exists() and folder.is_dir():
                info["subfolders"][name] = _bundle_schema_entry(folder)

        peak_var = base / "peak_bc" / "var.csv"
        if peak_var.exists():
            pvar = pd.read_csv(peak_var, nrows=5)
            info["recommended_columns"] = {
                "gene_col": "nearestGene" if "nearestGene" in pvar.columns else guess_column(pvar, ["gene", "gene_symbol", "nearest_gene"]),
                "distance_col": "distToTSS" if "distToTSS" in pvar.columns else guess_column(pvar, ["distance", "dist_to_tss"]),
            }

        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)
