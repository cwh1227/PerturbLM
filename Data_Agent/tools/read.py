"""
Data reading tools.

Each tool reads one format and writes THREE standardized intermediate files:

    intermediate/cell_meta.parquet   — all cells, standardized columns (no QC applied yet)
    intermediate/gene_meta.parquet   — all genes, standardized columns
    intermediate/data_source.json    — {format, path, ...} for expression data

WORKFLOW FOR THE AGENT
──────────────────────
1. Call inspect_* to see all column names and sample values.
2. Decide:
     - column_map : which source columns → 15 standard metadata columns
     - qc_col / qc_val : which column + value marks a passing cell/sample
       (leave empty to accept all rows; apply numeric thresholds later via
        qc_filter_cells)
3. Call read_* with those decisions.

Every downstream tool (qc_filter_cells, select_hvg, map_to_hgnc, tokenize)
reads only the standardized columns — completely format-agnostic.

STANDARDIZED SCHEMA WRITTEN BY EVERY read_* TOOL
─────────────────────────────────────────────────
cell_meta.parquet
  sample_id  (index)
  qc_pass    bool     — True = passes the dataset's own QC flag
  n_genes    float    — detected genes   (NaN for bulk data)
  n_counts   float    — total counts     (NaN for bulk data)
  pct_mt     float    — % mitochondrial  (NaN for bulk data)
  tissue region sampleType compartment
  cellType1 cellType2 majorCluster subCluster
  sex age perturbType disease drug dose time
  ... (all original columns kept for reference)

gene_meta.parquet
  orig_id  gene_symbol  ensembl_id  gctx_index  orig_token_id
  is_hvg   hgnc_num     hgnc_token
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

# Standard 15 metadata columns required in every cell_meta.parquet
_STD_META = [
    "tissue", "region", "sampleType", "compartment",
    "cellType1", "cellType2", "majorCluster", "subCluster",
    "sex", "age", "perturbType", "disease", "drug", "dose", "time",
]


def _resolve_dataset_split(ds, preferred_split: str):
    if hasattr(ds, "keys"):
        split_name = preferred_split if preferred_split in ds else list(ds.keys())[0]
        return ds[split_name]
    return ds


def _apply_column_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Copy source columns into standard metadata column names."""
    for std_col, src_col in col_map.items():
        if src_col and src_col in df.columns:
            df[std_col] = df[src_col].astype(str).fillna("")
    return df


def _fill_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all 15 standard metadata columns exist (empty string default)."""
    for col in _STD_META:
        if col not in df.columns:
            df[col] = ""
    return df


def _write_intermediate(
    intermediate_dir: Path,
    cell_meta: pd.DataFrame,
    gene_meta: pd.DataFrame,
    data_source: dict,
) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    cell_meta.to_parquet(intermediate_dir / "cell_meta.parquet", index=True)
    gene_meta.to_parquet(intermediate_dir / "gene_meta.parquet", index=False)
    (intermediate_dir / "data_source.json").write_text(
        json.dumps(data_source, indent=2)
    )


# ---------------------------------------------------------------------------
# 1. AnnData h5ad
# ---------------------------------------------------------------------------

@tool
def read_h5ad(
    path:             str,
    column_map:       str = "{}",
    qc_col:           str = "",
    qc_val:           str = "",
    intermediate_dir: str = "",
) -> str:
    """
    Read an AnnData .h5ad file and write standardized intermediate files.

    ALWAYS call inspect_h5ad first, then fill in column_map based on the obs
    columns you find.

    column_map — JSON dict mapping standard column names → obs column names.
                 Example: {"tissue": "tissue_general", "cellType1": "cell_type",
                           "sex": "gender", "disease": "condition",
                           "drug": "perturbation"}
                 Only map columns you found in inspect output.
                 Any unmapped standard columns → empty string.

    qc_col / qc_val — obs column and string value marking a passing cell.
                      Example: qc_col="quality_label", qc_val="good"
                      Leave empty if no pre-computed QC flag exists; apply
                      numeric thresholds via qc_filter_cells() instead.

    Auto-detected QC metrics (checked regardless of column_map):
      n_genes  ← n_genes_by_counts / n_genes
      n_counts ← total_counts / n_counts
      pct_mt   ← pct_counts_mt / pct_mt

    Writes to: {intermediate_dir}/ (default: {file_dir}/intermediate/)
    Next steps: qc_filter_cells → select_hvg → map_to_hgnc → tokenize
    """
    try:
        import anndata as ad

        p       = Path(path)
        idir    = Path(intermediate_dir) if intermediate_dir else p.parent / "intermediate"
        adata   = ad.read_h5ad(path, backed="r")
        col_map: dict = json.loads(column_map) if column_map.strip() not in ("", "{}") else {}

        # ── cell_meta ──────────────────────────────────────────────────────
        cell_meta = adata.obs.copy()
        cell_meta.index.name = "sample_id"

        # QC pass flag
        if qc_col and qc_col in cell_meta.columns:
            cell_meta["qc_pass"] = cell_meta[qc_col].astype(str) == str(qc_val)
        else:
            cell_meta["qc_pass"] = True

        # Auto-detect numeric QC metrics
        def _auto_num(srcs: list, dst: str) -> None:
            for s in srcs:
                if s in cell_meta.columns and dst not in cell_meta.columns:
                    cell_meta[dst] = pd.to_numeric(cell_meta[s], errors="coerce")
                    return
            if dst not in cell_meta.columns:
                cell_meta[dst] = float("nan")

        _auto_num(["n_genes_by_counts", "n_genes"], "n_genes")
        _auto_num(["total_counts", "n_counts"],     "n_counts")
        _auto_num(["pct_counts_mt", "pct_mt"],      "pct_mt")

        # Apply LLM-supplied column_map, then fill missing standard columns
        cell_meta = _apply_column_map(cell_meta, col_map)
        cell_meta = _fill_meta(cell_meta)

        # ── gene_meta ──────────────────────────────────────────────────────
        orig_ids  = adata.var_names.tolist()
        gene_meta = pd.DataFrame({
            "orig_id":       orig_ids,
            "gene_symbol":   orig_ids,
            "ensembl_id":    "",
            "gctx_index":    -1,
            "orig_token_id": -1,
            "is_hvg":        True,
            "hgnc_num":      "",
            "hgnc_token":    0,
        })
        for col in ("ensembl_id", "gene_ids", "Ensembl"):
            if col in adata.var.columns:
                gene_meta["ensembl_id"] = adata.var[col].fillna("").astype(str).values
                break

        adata.file.close()

        _write_intermediate(idir, cell_meta, gene_meta,
                            {"format": "h5ad", "path": str(p)})

        return json.dumps({
            "status":           "ok",
            "n_cells":          len(cell_meta),
            "n_genes":          len(gene_meta),
            "column_map_used":  col_map,
            "qc_col_used":      qc_col or "(none — qc_pass=True for all cells)",
            "intermediate_dir": str(idir),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 2. L1000 gctx + metadata TSVs
# ---------------------------------------------------------------------------

@tool
def read_gctx(
    gctx:                 str,
    siginfo:              str,
    cellinfo:             str,
    compoundinfo:         str,
    geneinfo:             str,
    column_map:           str = "{}",
    sig_id_col:           str = "",
    qc_col:               str = "",
    qc_val:               str = "",
    cellinfo_join_col:    str = "",
    compoundinfo_join_col: str = "",
    intermediate_dir:     str = "",
) -> str:
    """
    Read L1000/CLUE gctx + metadata TSV files and write standardized intermediate files.

    ALWAYS call inspect_gctx + inspect_tsv on siginfo / cellinfo / compoundinfo /
    geneinfo first, then fill in the parameters below based on what you find.

    column_map  — JSON dict mapping standard column names → merged siginfo column names.
                  After merging cellinfo and compoundinfo, all their columns are
                  available. Example:
                  {"tissue": "cell_iname", "drug": "cmap_name",
                   "dose": "pert_dose", "time": "pert_time",
                   "perturbType": "pert_type", "sampleType": "pert_type",
                   "sex": "donor_sex", "age": "donor_age"}

    sig_id_col  — column in siginfo to use as sample_id index (e.g. "sig_id",
                  "inst_id"). Auto-detected from common names if left empty.

    qc_col / qc_val — column and value in siginfo that marks a passing signature.
                      Example: qc_col="qc_pass", qc_val="1"
                      Leave empty to accept all rows.

    cellinfo_join_col     — join key between siginfo and cellinfo (e.g. "cell_iname").
                            Auto-detected if empty.
    compoundinfo_join_col — join key between siginfo and compoundinfo (e.g. "pert_id").
                            Auto-detected if empty.

    n_genes / n_counts / pct_mt — always NaN (bulk L1000 landmark data).

    Writes to: {gctx_dir}/intermediate/ (or intermediate_dir if specified)
    Next steps: qc_filter_cells → (select_hvg optional) → map_to_hgnc → tokenize
    """
    try:
        import h5py

        p       = Path(gctx)
        idir    = Path(intermediate_dir) if intermediate_dir else p.parent / "intermediate"
        col_map: dict = json.loads(column_map) if column_map.strip() not in ("", "{}") else {}

        # ── siginfo ────────────────────────────────────────────────────────
        sig = pd.read_csv(siginfo, sep="\t", dtype=str, low_memory=False)

        # Merge cellinfo
        cell_df = pd.read_csv(cellinfo, sep="\t", dtype=str, low_memory=False)
        join_cell = cellinfo_join_col or next(
            (c for c in sig.columns if c in cell_df.columns and c != sig.columns[0]),
            None
        )
        if join_cell and join_cell in cell_df.columns and join_cell in sig.columns:
            extra = [c for c in cell_df.columns if c != join_cell and c not in sig.columns]
            sig = sig.merge(cell_df[[join_cell] + extra].drop_duplicates(join_cell),
                            on=join_cell, how="left")

        # Merge compoundinfo
        cmpd = pd.read_csv(compoundinfo, sep="\t", dtype=str, low_memory=False)
        join_cmpd = compoundinfo_join_col or next(
            (c for c in sig.columns if c in cmpd.columns and c != sig.columns[0]),
            None
        )
        if join_cmpd and join_cmpd in cmpd.columns and join_cmpd in sig.columns:
            extra = [c for c in cmpd.columns if c != join_cmpd and c not in sig.columns]
            sig = sig.merge(cmpd[[join_cmpd] + extra].drop_duplicates(join_cmpd),
                            on=join_cmpd, how="left")

        # ── Standardize QC ─────────────────────────────────────────────────
        if qc_col and qc_col in sig.columns:
            sig["qc_pass"] = sig[qc_col].astype(str) == str(qc_val)
        else:
            sig["qc_pass"] = True

        sig["n_genes"]  = float("nan")
        sig["n_counts"] = float("nan")
        sig["pct_mt"]   = float("nan")

        # Apply LLM-supplied column_map, then fill missing standard columns
        sig = _apply_column_map(sig, col_map)
        sig = _fill_meta(sig)

        id_col = sig_id_col or next(
            (c for c in ("sig_id", "inst_id") if c in sig.columns), sig.columns[0]
        )
        sig.index = sig[id_col].astype(str)
        sig.index.name = "sample_id"

        # ── gene_meta from geneinfo ─────────────────────────────────────────
        gi      = pd.read_csv(geneinfo, sep="\t", dtype=str, low_memory=False)
        id_g    = next((c for c in ("gene_id", "entrez_id")          if c in gi.columns), gi.columns[0])
        sym_col = next((c for c in ("gene_symbol", "symbol")         if c in gi.columns), None)
        ens_col = next((c for c in ("ensembl_id", "Ensembl_Gene_ID") if c in gi.columns), None)

        gene_meta = pd.DataFrame({
            "orig_id":       gi[id_g].astype(str).tolist(),
            "gene_symbol":   gi[sym_col].astype(str).tolist() if sym_col else [""] * len(gi),
            "ensembl_id":    gi[ens_col].astype(str).tolist() if ens_col else [""] * len(gi),
            "gctx_index":    list(range(len(gi))),
            "orig_token_id": -1,
            "is_hvg":        True,
            "hgnc_num":      "",
            "hgnc_token":    0,
        })

        with h5py.File(gctx, "r") as f:
            shape = list(f["/0/DATA/0/matrix"].shape)

        _write_intermediate(idir, sig, gene_meta,
                            {"format": "gctx", "path": str(p)})

        return json.dumps({
            "status":           "ok",
            "n_signatures":     len(sig),
            "n_qc_pass":        int(sig["qc_pass"].sum()),
            "n_genes":          len(gene_meta),
            "gctx_shape":       shape,
            "column_map_used":  col_map,
            "qc_col_used":      qc_col or "(none — qc_pass=True for all)",
            "intermediate_dir": str(idir),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 3. HuggingFace Arrow dataset (Tahoe style)
# ---------------------------------------------------------------------------

@tool
def read_hf_dataset(
    dataset_dir:          str,
    column_map:           str = "{}",
    qc_col:               str = "",
    qc_val:               str = "",
    subset_split:         str = "train",
    obs_subset:           str = "obs_metadata",
    gene_subset:          str = "gene_metadata",
    cell_line_subset:     str = "cell_line_metadata",
    cell_line_join_col:   str = "",
    sample_id_col:        str = "",
    intermediate_dir:     str = "",
) -> str:
    """
    Read a HuggingFace Arrow dataset and write standardized intermediate files.

    ALWAYS call inspect_hf_dataset first to see the available subsets and their
    columns, then fill in the parameters below.

    column_map  — JSON dict mapping standard column names → obs / merged column names.
                  After merging cell_line_metadata, all its columns are available.
                  Example: {"tissue": "Organ", "cellType1": "cellType",
                            "disease": "pert_disease", "drug": "drug",
                            "dose": "dose", "perturbType": "pert_type"}

    qc_col / qc_val — column and value in obs that marks a passing cell.
                      Example: qc_col="pass_filter", qc_val="full"
                      Leave empty to accept all rows.

    subset_split         — HF split to use (default "train").
    obs_subset           — name of the obs metadata subset (default "obs_metadata").
    gene_subset          — name of the gene metadata subset (default "gene_metadata").
    cell_line_subset     — name of the cell line metadata subset to merge
                           (default "cell_line_metadata"). Set to "" to skip.
    cell_line_join_col   — join key for cell_line merge (auto-detected if empty).
    sample_id_col        — column to use as sample_id index (auto-detected if empty).

    Auto-detected QC metrics from obs columns (NaN if absent):
      n_genes  ← n_genes_by_counts / n_genes / num_genes
      n_counts ← total_counts / n_counts / num_counts
      pct_mt   ← pct_counts_mt / pct_mt

    Writes to: {dataset_dir}/intermediate/ (or intermediate_dir if specified)
    Next steps: qc_filter_cells → select_hvg → map_to_hgnc → tokenize
    """
    try:
        from datasets import load_from_disk

        root    = Path(dataset_dir)
        idir    = Path(intermediate_dir) if intermediate_dir else root / "intermediate"
        col_map: dict = json.loads(column_map) if column_map.strip() not in ("", "{}") else {}

        # ── obs_metadata ───────────────────────────────────────────────────
        obs_ds = _resolve_dataset_split(load_from_disk(str(root / obs_subset)), subset_split)
        obs    = obs_ds.to_pandas()

        # Merge cell_line_metadata (optional)
        if cell_line_subset:
            try:
                cl_ds  = _resolve_dataset_split(load_from_disk(str(root / cell_line_subset)), subset_split)
                cl_df  = cl_ds.to_pandas()
                join_cl = cell_line_join_col or next(
                    (c for c in obs.columns if c in cl_df.columns), None
                )
                if join_cl:
                    extra = [c for c in cl_df.columns if c != join_cl and c not in obs.columns]
                    obs = obs.merge(cl_df[[join_cl] + extra].drop_duplicates(join_cl),
                                   on=join_cl, how="left")
            except Exception:
                pass  # cell_line_metadata is optional

        # ── Standardize QC ─────────────────────────────────────────────────
        if qc_col and qc_col in obs.columns:
            obs["qc_pass"] = obs[qc_col].astype(str) == str(qc_val)
        else:
            obs["qc_pass"] = True

        # Auto-detect numeric QC metrics
        def _auto_num(names: list, dst: str) -> None:
            for n in names:
                if n in obs.columns and dst not in obs.columns:
                    obs[dst] = pd.to_numeric(obs[n], errors="coerce")
                    return
            if dst not in obs.columns:
                obs[dst] = float("nan")

        _auto_num(["n_genes_by_counts", "n_genes", "num_genes"], "n_genes")
        _auto_num(["total_counts", "n_counts", "num_counts"],    "n_counts")
        _auto_num(["pct_counts_mt", "pct_mt"],                   "pct_mt")

        # Apply LLM-supplied column_map, then fill missing standard columns
        obs = _apply_column_map(obs, col_map)
        obs = _fill_meta(obs)

        id_col = sample_id_col or next(
            (c for c in ("BARCODE_SUB_LIB_ID", "barcode", "cell_id") if c in obs.columns),
            obs.columns[0]
        )
        obs.index = obs[id_col].astype(str)
        obs.index.name = "sample_id"

        # ── gene_meta ──────────────────────────────────────────────────────
        gene_ds  = _resolve_dataset_split(load_from_disk(str(root / gene_subset)), subset_split)
        gene_df  = gene_ds.to_pandas()

        tok_col = next((c for c in ("token_id", "gene_id") if c in gene_df.columns),
                       gene_df.columns[0])
        gene_meta = pd.DataFrame({
            "orig_id":       gene_df[tok_col].astype(str).tolist(),
            "gene_symbol":   gene_df.get("gene_symbol",
                                         pd.Series([""] * len(gene_df))).astype(str).tolist(),
            "ensembl_id":    gene_df.get("ensembl_id",
                                         pd.Series([""] * len(gene_df))).astype(str).tolist(),
            "gctx_index":    -1,
            "orig_token_id": gene_df[tok_col].astype(int).tolist(),
            "is_hvg":        True,
            "hgnc_num":      "",
            "hgnc_token":    0,
        })

        _write_intermediate(idir, obs, gene_meta, {
            "format":       "hf",
            "path":         str(root),
            "subset_split": subset_split,
        })

        return json.dumps({
            "status":           "ok",
            "n_cells":          len(obs),
            "n_qc_pass":        int(obs["qc_pass"].sum()),
            "n_genes":          len(gene_meta),
            "column_map_used":  col_map,
            "qc_col_used":      qc_col or "(none — qc_pass=True for all cells)",
            "intermediate_dir": str(idir),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
