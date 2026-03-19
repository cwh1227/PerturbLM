"""
ATAC aggregate tool — aggregate peak-level data to gene-level scores.

Tool:
  aggregate_peaks_to_genes — reads intermediate files, writes final gene-score outputs.

For bundle mode: loads the sparse matrix, applies QC cell mask, then computes
per-peak openness as the sum over QC-passing cells.
For flat table mode: uses the openness column already stored in peak_gene_link.parquet.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool


@tool
def aggregate_peaks_to_genes(
    intermediate_dir: str,
    out_dir: str,
    use_absolute_distance: bool = True,
    min_peaks_per_gene: int = 1,
    top_n: int = 50,
) -> str:
    """
    Aggregate peak-level ATAC data to gene-level scores.

    Reads from intermediate_dir:
      peak_gene_link.parquet  — required; produced by read_atac_table or read_atac_bundle
      data_source.json        — required; determines mode and matrix path
      cell_meta.parquet       — bundle mode only; provides QC mask for openness

    Bundle mode:  loads counts.mtx.gz, applies QC-passing cell mask,
                  computes per-peak openness = sum of reads across QC cells.
    Flat table:   uses openness column already in peak_gene_link.parquet.

    Gene-level output columns:
      gene, n_peaks, mean_distance, median_distance, mean_abs_distance,
      composite_score (= mean(|distance|) by default),
      weighted_composite_score (openness-weighted; present when openness is available),
      rank

    Outputs in out_dir:
      atac_gene_scores.tsv
      atac_gene_scores.parquet
      atac_gene_scores_summary.json

    Args:
      intermediate_dir:      Directory with intermediate files from read_atac_*
      out_dir:               Output directory
      use_absolute_distance: Use |distance| before averaging (default True)
      min_peaks_per_gene:    Minimum peaks required to include a gene (default 1)
      top_n:                 Number of top genes included in preview (default 50)
    """
    try:
        int_path = Path(intermediate_dir)
        ds_file = int_path / "data_source.json"
        link_file = int_path / "peak_gene_link.parquet"

        if not link_file.exists():
            return json.dumps(
                {"error": f"peak_gene_link.parquet not found in {intermediate_dir}"},
                ensure_ascii=False,
            )
        if not ds_file.exists():
            return json.dumps(
                {"error": f"data_source.json not found in {intermediate_dir}"},
                ensure_ascii=False,
            )

        ds = json.loads(ds_file.read_text(encoding="utf-8"))
        link = pd.read_parquet(link_file)
        link = link[link["is_valid_link"]].copy()

        openness_source_desc = ""

        # ── resolve openness ──
        if ds.get("format") == "atac_bundle" and ds.get("has_cell_meta"):
            cell_file = int_path / "cell_meta.parquet"
            if not cell_file.exists():
                return json.dumps(
                    {"error": "cell_meta.parquet not found for bundle mode."},
                    ensure_ascii=False,
                )
            cell_meta = pd.read_parquet(cell_file)
            qc_mask = (
                cell_meta["qc_pass"].values
                if "qc_pass" in cell_meta.columns
                else np.ones(len(cell_meta), dtype=bool)
            )
            n_pass = int(qc_mask.sum())
            n_total = int(len(cell_meta))

            from scipy.io import mmread
            matrix = mmread(ds["matrix_path"]).tocsr()
            filtered = matrix[qc_mask, :]
            peak_openness = np.asarray(filtered.sum(axis=0)).ravel()

            link["openness"] = peak_openness[link["source_index"].values]
            openness_source_desc = f"QC-filtered matrix ({n_pass}/{n_total} cells)"
        else:
            openness_source_desc = ds.get("openness_col_used") or "none"

        # ── scoring ──
        if use_absolute_distance:
            link["distance_for_score"] = link["distance_to_tss"].abs()
        else:
            link["distance_for_score"] = link["distance_to_tss"]

        has_openness = link["openness"].notna().any()

        grouped = link.groupby("gene_symbol", dropna=False)
        out = grouped.agg(
            n_peaks=("distance_to_tss", "size"),
            mean_distance=("distance_to_tss", "mean"),
            median_distance=("distance_to_tss", "median"),
            mean_abs_distance=("distance_to_tss", lambda x: float(np.mean(np.abs(x)))),
            composite_score=("distance_for_score", "mean"),
        ).reset_index().rename(columns={"gene_symbol": "gene"})

        if has_openness:
            weighted_vals = []
            for gene, sub in grouped:
                w = pd.to_numeric(sub["openness"], errors="coerce").fillna(0.0).values
                x = sub["distance_for_score"].values
                if np.sum(w) > 0:
                    weighted_vals.append((gene, float(np.average(x, weights=w))))
                else:
                    weighted_vals.append((gene, float(np.mean(x))))
            wdf = pd.DataFrame(weighted_vals, columns=["gene", "weighted_composite_score"])
            out = out.merge(wdf, on="gene", how="left")

        out = out[out["n_peaks"] >= int(max(1, min_peaks_per_gene))].copy()
        out = out.sort_values(["composite_score", "n_peaks"], ascending=[True, False]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)

        result_path = Path(out_dir)
        result_path.mkdir(parents=True, exist_ok=True)

        tsv_file = result_path / "atac_gene_scores.tsv"
        parquet_file = result_path / "atac_gene_scores.parquet"
        summary_file = result_path / "atac_gene_scores_summary.json"

        out.to_csv(tsv_file, sep="\t", index=False)
        out.to_parquet(parquet_file, index=False)

        preview_cols = [c for c in ["rank", "gene", "composite_score", "weighted_composite_score", "n_peaks"] if c in out.columns]
        summary = {
            "status": "ok",
            "source_format": ds.get("format", "unknown"),
            "n_genes": int(len(out)),
            "openness_source": openness_source_desc,
            "score_definition": (
                "composite_score = mean(|distance|)"
                if use_absolute_distance
                else "composite_score = mean(distance)"
            ),
            "outputs": {
                "tsv": str(tsv_file),
                "parquet": str(parquet_file),
                "summary": str(summary_file),
            },
            "top_genes_preview": out.head(max(1, top_n))[preview_cols].to_dict(orient="records"),
        }
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)
