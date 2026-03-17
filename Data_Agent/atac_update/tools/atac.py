"""
ATAC key-feature extraction tools.

Goal:
- Aggregate peak-level ATAC records into gene-level scores.
- Primary score (composite_score): average distance from open chromatin peaks to gene.

Typical input table schema (any delimiter / parquet supported):
  - gene column:      gene / gene_symbol / nearest_gene / target_gene / symbol ...
  - distance column:  distance / dist_to_tss / tss_distance / peak_to_gene_distance ...
  - openness column (optional): score / accessibility / signal / peak_score ...
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool


_GENE_CANDIDATES = [
    "gene",
    "gene_symbol",
    "symbol",
    "nearest_gene",
    "target_gene",
    "gene_name",
    "hgnc_symbol",
]

_DISTANCE_CANDIDATES = [
    "distance",
    "dist_to_tss",
    "tss_distance",
    "peak_to_gene_distance",
    "distance_to_gene",
    "dist",
]

_OPENNESS_CANDIDATES = [
    "score",
    "accessibility",
    "open_score",
    "signal",
    "peak_score",
    "counts",
]


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".csv"}:
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, sep=None, engine="python")


def _guess_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_to_real:
            return lower_to_real[c]
    return ""


def _read_mtx_shape_gz(mtx_gz: Path) -> tuple[int, int, int | None]:
    import gzip

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
    Inspect an ATAC peak-to-gene table and suggest key columns.

    Returns:
      - columns
      - sample rows
      - guessed_gene_col
      - guessed_distance_col
      - guessed_openness_col (optional)
    """
    try:
        df = _read_table(path)
        gene_col = _guess_column(df, _GENE_CANDIDATES)
        distance_col = _guess_column(df, _DISTANCE_CANDIDATES)
        openness_col = _guess_column(df, _OPENNESS_CANDIDATES)

        sample = df.head(max(1, nrows)).replace({np.nan: None}).to_dict(orient="records")
        return json.dumps(
            {
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "columns": [str(c) for c in df.columns],
                "guessed_gene_col": gene_col,
                "guessed_distance_col": distance_col,
                "guessed_openness_col": openness_col,
                "sample_rows": sample,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


@tool
def compute_atac_gene_scores(
    path: str,
    out_dir: str,
    gene_col: str = "",
    distance_col: str = "",
    openness_col: str = "",
    use_absolute_distance: bool = True,
    min_peaks_per_gene: int = 1,
    top_n: int = 50,
) -> str:
    """
    Compute ATAC gene-level key scores from a peak-to-gene table.

    Core requirement implemented:
      composite_score = average peak-to-gene distance for each gene
      (absolute average by default: use_absolute_distance=True)

    Args:
      path: Input table path (.csv/.tsv/.txt/.parquet)
      out_dir: Output directory
      gene_col: Gene symbol column. If empty, auto-detect.
      distance_col: Peak-to-gene distance column. If empty, auto-detect.
      openness_col: Optional openness/intensity column for weighted stats.
      use_absolute_distance: Use |distance| before averaging.
      min_peaks_per_gene: Keep genes with at least this many peaks.
      top_n: Number of top genes (smallest composite_score) in preview.

    Outputs in out_dir:
      - atac_gene_scores.tsv
      - atac_gene_scores.parquet
      - atac_gene_scores_summary.json
    """
    try:
        df = _read_table(path)
        if df.empty:
            return json.dumps({"error": "Input table is empty."}, ensure_ascii=False)

        resolved_gene_col = gene_col or _guess_column(df, _GENE_CANDIDATES)
        resolved_distance_col = distance_col or _guess_column(df, _DISTANCE_CANDIDATES)
        resolved_openness_col = openness_col or _guess_column(df, _OPENNESS_CANDIDATES)

        if not resolved_gene_col:
            return json.dumps(
                {
                    "error": "Cannot determine gene column. Please pass gene_col explicitly.",
                    "columns": [str(c) for c in df.columns],
                },
                ensure_ascii=False,
            )

        if not resolved_distance_col:
            return json.dumps(
                {
                    "error": "Cannot determine distance column. Please pass distance_col explicitly.",
                    "columns": [str(c) for c in df.columns],
                },
                ensure_ascii=False,
            )

        work = df[[resolved_gene_col, resolved_distance_col] + ([resolved_openness_col] if resolved_openness_col else [])].copy()
        work = work.rename(columns={resolved_gene_col: "gene", resolved_distance_col: "distance"})
        work["gene"] = work["gene"].astype(str).str.strip()
        work["distance"] = pd.to_numeric(work["distance"], errors="coerce")
        work = work[work["gene"] != ""]
        work = work.dropna(subset=["distance"])

        if use_absolute_distance:
            work["distance_for_score"] = work["distance"].abs()
        else:
            work["distance_for_score"] = work["distance"]

        if resolved_openness_col:
            work[resolved_openness_col] = pd.to_numeric(work[resolved_openness_col], errors="coerce").fillna(0.0)

        grouped = work.groupby("gene", dropna=False)
        out = grouped.agg(
            n_peaks=("distance", "size"),
            mean_distance=("distance", "mean"),
            median_distance=("distance", "median"),
            mean_abs_distance=("distance", lambda x: float(np.mean(np.abs(x)))),
            composite_score=("distance_for_score", "mean"),
        ).reset_index()

        if resolved_openness_col:
            weighted_vals = []
            for gene, sub in grouped:
                w = pd.to_numeric(sub[resolved_openness_col], errors="coerce").fillna(0.0).values
                x = sub["distance_for_score"].values
                if np.sum(w) > 0:
                    weighted_vals.append((gene, float(np.average(x, weights=w))))
                else:
                    weighted_vals.append((gene, float(np.mean(x))))
            weighted_df = pd.DataFrame(weighted_vals, columns=["gene", "weighted_composite_score"])
            out = out.merge(weighted_df, on="gene", how="left")

        out = out[out["n_peaks"] >= int(max(1, min_peaks_per_gene))].copy()
        out = out.sort_values(["composite_score", "n_peaks"], ascending=[True, False]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        tsv_file = out_path / "atac_gene_scores.tsv"
        parquet_file = out_path / "atac_gene_scores.parquet"
        summary_file = out_path / "atac_gene_scores_summary.json"

        out.to_csv(tsv_file, sep="\t", index=False)
        out.to_parquet(parquet_file, index=False)

        preview_df = out.head(max(1, top_n))
        summary = {
            "status": "ok",
            "input_path": str(path),
            "n_input_rows": int(len(df)),
            "n_valid_rows": int(len(work)),
            "n_genes": int(len(out)),
            "gene_col_used": resolved_gene_col,
            "distance_col_used": resolved_distance_col,
            "openness_col_used": resolved_openness_col or "",
            "score_definition": "composite_score = mean(|distance|)" if use_absolute_distance else "composite_score = mean(distance)",
            "outputs": {
                "tsv": str(tsv_file),
                "parquet": str(parquet_file),
                "summary": str(summary_file),
            },
            "top_genes_preview": preview_df[[c for c in ["rank", "gene", "composite_score", "n_peaks"] if c in preview_df.columns]].to_dict(orient="records"),
        }
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)


@tool
def inspect_atac_bundle(dataset_dir: str) -> str:
    """
    Inspect a directory-form scATAC dataset bundle.

    Expected subfolders (when present):
      - peak_bc
      - gene_scores
      - ChromVar
      - LSI_embedding
      - markerpeak_target

    Each subfolder is inspected for:
      counts.mtx.gz shape, obs.csv columns/rows, var.csv columns/rows.
    """
    try:
        base = Path(dataset_dir)
        if not base.exists():
            return json.dumps({"error": f"dataset_dir not found: {dataset_dir}"}, ensure_ascii=False)

        names = ["peak_bc", "gene_scores", "ChromVar", "LSI_embedding", "markerpeak_target"]
        info = {
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
            info["recommended_columns_for_gene_score"] = {
                "gene_col": "nearestGene" if "nearestGene" in pvar.columns else "",
                "distance_col": "distToTSS" if "distToTSS" in pvar.columns else "",
                "openness_col": "ncounts" if "ncounts" in pvar.columns else "",
            }

        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)


@tool
def compute_atac_gene_scores_from_bundle(
    dataset_dir: str,
    out_dir: str,
    gene_col: str = "nearestGene",
    distance_col: str = "distToTSS",
    openness_col_fallback: str = "ncounts",
    use_absolute_distance: bool = True,
    min_peaks_per_gene: int = 1,
    min_tss_enrichment: float = 4.0,
    min_frip: float = 0.15,
    min_fragments: int = 3000,
    max_blacklist_ratio: float = 0.05,
    min_peak_cells: int = 1,
    top_n: int = 50,
) -> str:
    """
    Compute gene-level ATAC scores directly from a bundle directory with QC.

    Required files:
      peak_bc/obs.csv, peak_bc/var.csv, peak_bc/counts.mtx.gz

    QC (cell-level, on peak_bc/obs.csv):
      TSSEnrichment >= min_tss_enrichment
      FRIP >= min_frip
      nFrags >= min_fragments
      BlacklistRatio <= max_blacklist_ratio

    Score:
      composite_score = mean(|distance|) by default per gene
      weighted_composite_score uses peak openness from QC-passing cells.
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
            return json.dumps({"error": f"gene_col '{gene_col}' not found in peak_bc/var.csv", "columns": [str(c) for c in var.columns]}, ensure_ascii=False)
        if distance_col not in var.columns:
            return json.dumps({"error": f"distance_col '{distance_col}' not found in peak_bc/var.csv", "columns": [str(c) for c in var.columns]}, ensure_ascii=False)

        qc_mask = pd.Series([True] * len(obs))
        qc_used = {}

        if "TSSEnrichment" in obs.columns:
            tsse = pd.to_numeric(obs["TSSEnrichment"], errors="coerce")
            qc_mask = qc_mask & (tsse >= float(min_tss_enrichment))
            qc_used["TSSEnrichment"] = {"op": ">=", "value": float(min_tss_enrichment)}

        if "FRIP" in obs.columns:
            frip = pd.to_numeric(obs["FRIP"], errors="coerce")
            qc_mask = qc_mask & (frip >= float(min_frip))
            qc_used["FRIP"] = {"op": ">=", "value": float(min_frip)}

        if "nFrags" in obs.columns:
            nfrags = pd.to_numeric(obs["nFrags"], errors="coerce")
            qc_mask = qc_mask & (nfrags >= int(min_fragments))
            qc_used["nFrags"] = {"op": ">=", "value": int(min_fragments)}

        if "BlacklistRatio" in obs.columns:
            blk = pd.to_numeric(obs["BlacklistRatio"], errors="coerce")
            qc_mask = qc_mask & (blk <= float(max_blacklist_ratio))
            qc_used["BlacklistRatio"] = {"op": "<=", "value": float(max_blacklist_ratio)}

        n_cells_total = int(len(obs))
        n_cells_pass_qc = int(qc_mask.sum())
        if n_cells_pass_qc <= 0:
            return json.dumps({"error": "No cells pass QC thresholds.", "qc": qc_used}, ensure_ascii=False)

        from scipy.io import mmread

        matrix = mmread(str(mtx_file)).tocsr()
        if matrix.shape[0] != len(obs) or matrix.shape[1] != len(var):
            return json.dumps(
                {
                    "error": "Matrix shape does not match obs/var rows.",
                    "matrix_shape": list(matrix.shape),
                    "obs_rows": int(len(obs)),
                    "var_rows": int(len(var)),
                },
                ensure_ascii=False,
            )

        filtered = matrix[qc_mask.values, :]
        peak_open = np.asarray(filtered.sum(axis=0)).ravel()

        work = pd.DataFrame(
            {
                "gene": var[gene_col].astype(str).str.strip(),
                "distance": pd.to_numeric(var[distance_col], errors="coerce"),
                "open_score": peak_open,
                "peak_ncells": pd.to_numeric(var["ncells"], errors="coerce") if "ncells" in var.columns else np.nan,
            }
        )
        work = work[(work["gene"] != "") & (~work["distance"].isna())]
        if "peak_ncells" in work.columns:
            work = work[(work["peak_ncells"].isna()) | (work["peak_ncells"] >= int(max(1, min_peak_cells)))]

        if use_absolute_distance:
            work["distance_for_score"] = work["distance"].abs()
        else:
            work["distance_for_score"] = work["distance"]

        grouped = work.groupby("gene", dropna=False)
        out = grouped.agg(
            n_peaks=("distance", "size"),
            mean_distance=("distance", "mean"),
            median_distance=("distance", "median"),
            mean_abs_distance=("distance", lambda x: float(np.mean(np.abs(x)))),
            composite_score=("distance_for_score", "mean"),
            openness_sum_qc=("open_score", "sum"),
        ).reset_index()

        weighted_vals = []
        for gene, sub in grouped:
            w = pd.to_numeric(sub["open_score"], errors="coerce").fillna(0.0).values
            x = sub["distance_for_score"].values
            if np.sum(w) > 0:
                weighted_vals.append((gene, float(np.average(x, weights=w))))
            else:
                weighted_vals.append((gene, float(np.mean(x))))
        weighted_df = pd.DataFrame(weighted_vals, columns=["gene", "weighted_composite_score"])
        out = out.merge(weighted_df, on="gene", how="left")

        out = out[out["n_peaks"] >= int(max(1, min_peaks_per_gene))].copy()
        out = out.sort_values(["composite_score", "n_peaks"], ascending=[True, False]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        tsv_file = out_path / "atac_gene_scores.tsv"
        parquet_file = out_path / "atac_gene_scores.parquet"
        summary_file = out_path / "atac_gene_scores_summary.json"

        out.to_csv(tsv_file, sep="\t", index=False)
        out.to_parquet(parquet_file, index=False)

        preview_df = out.head(max(1, top_n))
        summary = {
            "status": "ok",
            "dataset_dir": str(dataset_dir),
            "source": "peak_bc",
            "n_cells_total": n_cells_total,
            "n_cells_pass_qc": n_cells_pass_qc,
            "n_peaks_total": int(len(var)),
            "n_peaks_used": int(len(work)),
            "n_genes": int(len(out)),
            "gene_col_used": gene_col,
            "distance_col_used": distance_col,
            "openness_source": "QC-filtered peak_bc/counts.mtx.gz" if n_cells_pass_qc < n_cells_total else f"peak_bc/{openness_col_fallback}",
            "score_definition": "composite_score = mean(|distance|)" if use_absolute_distance else "composite_score = mean(distance)",
            "qc_thresholds_used": qc_used,
            "outputs": {
                "tsv": str(tsv_file),
                "parquet": str(parquet_file),
                "summary": str(summary_file),
            },
            "top_genes_preview": preview_df[[c for c in ["rank", "gene", "composite_score", "weighted_composite_score", "n_peaks"] if c in preview_df.columns]].to_dict(orient="records"),
        }
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)
