"""
ATAC read tools — parse raw inputs and write standardized intermediate files.

Tools:
  read_atac_table   — flat peak-to-gene table → peak_gene_link.parquet + data_source.json
  read_atac_bundle  — scATAC bundle directory → cell_meta.parquet + peak_meta.parquet
                      + peak_gene_link.parquet + data_source.json

The sparse matrix (counts.mtx.gz) is NOT loaded here.
It is deferred to aggregate_peaks_to_genes, which applies the QC mask from
cell_meta.parquet before computing per-peak openness.
"""

from __future__ import annotations

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
    QC_COL_MAP,
)


@tool
def read_atac_table(
    path: str,
    intermediate_dir: str,
    gene_col: str = "",
    distance_col: str = "",
    openness_col: str = "",
) -> str:
    """
    Read a flat ATAC peak-to-gene table and write standardized intermediate files.

    Flat tables have no cell-level information, so only two files are written:
      peak_gene_link.parquet  — one row per peak:
                                peak_id, gene_symbol, distance_to_tss,
                                openness (NaN if unavailable),
                                link_type ("nearest"), is_valid_link, source_index
      data_source.json        — source metadata (format, paths, has_cell_meta=false)

    Args:
      path:             Input file (.csv/.tsv/.txt/.parquet)
      intermediate_dir: Directory to write intermediate files
      gene_col:         Gene symbol column name. Empty → auto-detect.
      distance_col:     Peak-to-TSS distance column name. Empty → auto-detect.
      openness_col:     Optional openness/intensity column. Empty → auto-detect.
    """
    try:
        df = read_table(path)
        if df.empty:
            return json.dumps({"error": "Input table is empty."}, ensure_ascii=False)

        resolved_gene_col = gene_col or guess_column(df, _GENE_CANDIDATES)
        resolved_distance_col = distance_col or guess_column(df, _DISTANCE_CANDIDATES)
        resolved_openness_col = openness_col or guess_column(df, _OPENNESS_CANDIDATES)

        if not resolved_gene_col:
            return json.dumps({"error": "Cannot determine gene column.", "columns": [str(c) for c in df.columns]}, ensure_ascii=False)
        if not resolved_distance_col:
            return json.dumps({"error": "Cannot determine distance column.", "columns": [str(c) for c in df.columns]}, ensure_ascii=False)

        link = pd.DataFrame({
            "peak_id": np.arange(len(df)),
            "gene_symbol": df[resolved_gene_col].astype(str).str.strip().values,
            "distance_to_tss": pd.to_numeric(df[resolved_distance_col], errors="coerce").values,
            "openness": (
                pd.to_numeric(df[resolved_openness_col], errors="coerce").values
                if resolved_openness_col
                else np.full(len(df), np.nan)
            ),
            "link_type": "nearest",
            "source_index": np.arange(len(df)),
        })
        link["is_valid_link"] = link["gene_symbol"].ne("") & link["distance_to_tss"].notna()

        out_path = Path(intermediate_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        link_file = out_path / "peak_gene_link.parquet"
        link.to_parquet(link_file, index=False)

        data_source = {
            "format": "atac_table",
            "path": str(path),
            "has_cell_meta": False,
            "has_peak_meta": False,
            "gene_col_used": resolved_gene_col,
            "distance_col_used": resolved_distance_col,
            "openness_col_used": resolved_openness_col or "",
            "n_peaks": int(len(df)),
        }
        (out_path / "data_source.json").write_text(
            json.dumps(data_source, indent=2), encoding="utf-8"
        )

        return json.dumps({
            "status": "ok",
            "n_peaks": int(len(df)),
            "n_valid_links": int(link["is_valid_link"].sum()),
            "gene_col_used": resolved_gene_col,
            "distance_col_used": resolved_distance_col,
            "openness_col_used": resolved_openness_col or "",
            "outputs": {
                "peak_gene_link": str(link_file),
                "data_source": str(out_path / "data_source.json"),
            },
        }, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)


@tool
def read_atac_bundle(
    dataset_dir: str,
    intermediate_dir: str,
    gene_col: str = "nearestGene",
    distance_col: str = "distToTSS",
) -> str:
    """
    Read a scATAC bundle directory and write standardized intermediate files.

    Reads peak_bc/obs.csv (cells) and peak_bc/var.csv (peaks).
    The sparse matrix (counts.mtx.gz) is NOT loaded here — it is deferred to
    aggregate_peaks_to_genes, which applies the QC mask after qc_filter_atac_cells.

    Writes to intermediate_dir:
      cell_meta.parquet      — per-cell QC metrics, qc_pass=True for all (pre-QC)
                               Standardized columns: cell_id, qc_pass,
                               tss_enrichment, frip, n_fragments, blacklist_ratio,
                               plus any extra obs columns preserved as-is.
      peak_meta.parquet      — all var.csv columns, plus peak_id (sequential index)
      peak_gene_link.parquet — peak-to-gene links, openness=NaN (computed after QC)
                               Columns: peak_id, gene_symbol, distance_to_tss,
                               openness, link_type, is_valid_link, source_index
      data_source.json       — source metadata including matrix_path

    Args:
      dataset_dir:      Bundle directory containing peak_bc/ subfolder
      intermediate_dir: Directory to write intermediate files
      gene_col:         Gene symbol column in peak_bc/var.csv (default: nearestGene)
      distance_col:     Distance-to-TSS column in peak_bc/var.csv (default: distToTSS)
    """
    try:
        base = Path(dataset_dir)
        peak_dir = base / "peak_bc"
        obs_file = peak_dir / "obs.csv"
        var_file = peak_dir / "var.csv"
        mtx_file = peak_dir / "counts.mtx.gz"

        missing = [str(p) for p in [obs_file, var_file, mtx_file] if not p.exists()]
        if missing:
            return json.dumps({"error": f"Missing required files: {missing}"}, ensure_ascii=False)

        obs = pd.read_csv(obs_file)
        var = pd.read_csv(var_file)

        if gene_col not in var.columns:
            return json.dumps({
                "error": f"gene_col '{gene_col}' not found in peak_bc/var.csv",
                "columns": [str(c) for c in var.columns],
            }, ensure_ascii=False)
        if distance_col not in var.columns:
            return json.dumps({
                "error": f"distance_col '{distance_col}' not found in peak_bc/var.csv",
                "columns": [str(c) for c in var.columns],
            }, ensure_ascii=False)

        # ── cell_meta: standardize QC column names ──
        cell_meta = pd.DataFrame()
        if "barcode" in obs.columns:
            cell_meta["cell_id"] = obs["barcode"].astype(str)
        else:
            cell_meta["cell_id"] = obs.index.astype(str)
        cell_meta["qc_pass"] = True  # all pass before qc_filter_atac_cells

        for src, dst in QC_COL_MAP.items():
            if src in obs.columns:
                cell_meta[dst] = pd.to_numeric(obs[src], errors="coerce")

        # preserve extra obs columns
        skip_cols = set(QC_COL_MAP) | {"barcode"}
        for c in obs.columns:
            if c not in skip_cols:
                cell_meta[c] = obs[c].values

        # ── peak_meta: var.csv + sequential peak_id ──
        peak_meta = var.copy()
        peak_meta.insert(0, "peak_id", np.arange(len(var)))

        # ── peak_gene_link: from var.csv, openness deferred ──
        link = pd.DataFrame({
            "peak_id": np.arange(len(var)),
            "gene_symbol": var[gene_col].astype(str).str.strip().values,
            "distance_to_tss": pd.to_numeric(var[distance_col], errors="coerce").values,
            "openness": np.full(len(var), np.nan),
            "link_type": "nearest",
            "source_index": np.arange(len(var)),
        })
        link["is_valid_link"] = link["gene_symbol"].ne("") & link["distance_to_tss"].notna()

        out_path = Path(intermediate_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cell_meta_file = out_path / "cell_meta.parquet"
        peak_meta_file = out_path / "peak_meta.parquet"
        link_file = out_path / "peak_gene_link.parquet"

        cell_meta.to_parquet(cell_meta_file, index=False)
        peak_meta.to_parquet(peak_meta_file, index=False)
        link.to_parquet(link_file, index=False)

        data_source = {
            "format": "atac_bundle",
            "dataset_dir": str(dataset_dir),
            "has_cell_meta": True,
            "has_peak_meta": True,
            "matrix_path": str(mtx_file),
            "gene_col_used": gene_col,
            "distance_col_used": distance_col,
            "n_cells": int(len(obs)),
            "n_peaks": int(len(var)),
        }
        (out_path / "data_source.json").write_text(
            json.dumps(data_source, indent=2), encoding="utf-8"
        )

        return json.dumps({
            "status": "ok",
            "n_cells": int(len(obs)),
            "n_peaks": int(len(var)),
            "n_valid_links": int(link["is_valid_link"].sum()),
            "cell_meta_columns": [str(c) for c in cell_meta.columns],
            "gene_col_used": gene_col,
            "distance_col_used": distance_col,
            "outputs": {
                "cell_meta": str(cell_meta_file),
                "peak_meta": str(peak_meta_file),
                "peak_gene_link": str(link_file),
                "data_source": str(out_path / "data_source.json"),
            },
        }, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)
