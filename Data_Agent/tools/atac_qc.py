"""
ATAC QC tool — filter cells in cell_meta.parquet by ATAC QC thresholds.

Tool:
  qc_filter_atac_cells — reads/writes cell_meta.parquet, updates qc_pass column.
                         Skipped automatically for flat table input (no cell-level data).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool


@tool
def qc_filter_atac_cells(
    intermediate_dir: str,
    min_tss_enrichment: float = 4.0,
    min_frip: float = 0.15,
    min_fragments: int = 3000,
    max_blacklist_ratio: float = 0.05,
) -> str:
    """
    Filter cells in cell_meta.parquet by ATAC QC thresholds.

    Updates the qc_pass column in cell_meta.parquet in place.
    Automatically skipped if data_source.json reports has_cell_meta=false
    (flat table input has no cell-level information).

    Standardized QC columns expected (absent columns are silently skipped):
      tss_enrichment   >= min_tss_enrichment
      frip             >= min_frip
      n_fragments      >= min_fragments
      blacklist_ratio  <= max_blacklist_ratio

    Args:
      intermediate_dir:     Directory containing cell_meta.parquet + data_source.json
      min_tss_enrichment:   Minimum TSS enrichment score (default 4.0)
      min_frip:             Minimum fraction of reads in peaks (default 0.15)
      min_fragments:        Minimum fragment count (default 3000)
      max_blacklist_ratio:  Maximum blacklist ratio (default 0.05)
    """
    try:
        out_path = Path(intermediate_dir)
        ds_file = out_path / "data_source.json"
        cell_file = out_path / "cell_meta.parquet"

        if ds_file.exists():
            ds = json.loads(ds_file.read_text(encoding="utf-8"))
            if not ds.get("has_cell_meta", True):
                return json.dumps({
                    "status": "skipped",
                    "reason": "has_cell_meta=false — flat table input has no cell-level QC.",
                }, ensure_ascii=False, indent=2)

        if not cell_file.exists():
            return json.dumps(
                {"error": f"cell_meta.parquet not found in {intermediate_dir}"},
                ensure_ascii=False,
            )

        cell_meta = pd.read_parquet(cell_file)
        n_total = int(len(cell_meta))
        qc_mask = pd.Series([True] * n_total, index=cell_meta.index)
        applied: dict = {}

        thresholds = {
            "tss_enrichment":  (">=", float(min_tss_enrichment)),
            "frip":            (">=", float(min_frip)),
            "n_fragments":     (">=", int(min_fragments)),
            "blacklist_ratio": ("<=", float(max_blacklist_ratio)),
        }
        for col, (op, val) in thresholds.items():
            if col in cell_meta.columns:
                series = pd.to_numeric(cell_meta[col], errors="coerce")
                qc_mask = qc_mask & (series >= val if op == ">=" else series <= val)
                applied[col] = {"op": op, "value": val}

        cell_meta["qc_pass"] = qc_mask.values
        cell_meta.to_parquet(cell_file, index=False)

        n_pass = int(qc_mask.sum())
        return json.dumps({
            "status": "ok",
            "n_cells_total": n_total,
            "n_cells_pass": n_pass,
            "n_cells_removed": n_total - n_pass,
            "thresholds_applied": applied,
        }, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)
