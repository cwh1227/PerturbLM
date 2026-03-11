"""
Metadata loading tools.

Three variants cover the most common input formats.
Each tool stores its result in STORE for subsequent pipeline steps.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from langchain_core.tools import tool

from Agent.state import STORE


# ---------------------------------------------------------------------------
# 1. AnnData h5ad  (single-cell — QC filter + HVG selection)
# ---------------------------------------------------------------------------

@tool
def load_metadata_h5ad(
    h5ad:             str,
    out_dir:          str,
    batch_key:        str   = "datasetID",
    n_top_genes:      int   = 3000,
    hvg_sample_total: int   = 200000,
    min_genes:        int   = 500,
    max_genes:        int   = 8000,
    min_counts:       int   = 1000,
    max_counts:       int   = 100000,
    max_pct_mt:       float = 20.0,
    mt_prefix:        str   = "MT-",
    qc_chunk_size:    int   = 50000,
    seed:             int   = 0,
) -> str:
    """
    Load single-cell AnnData, apply QC filtering, and select highly variable genes (HVG).

    Stores result in STORE['hvg_genes'] (original gene names) for the gene-vocabulary step.
    Outputs: out_dir/hvg_genes.txt

    Returns JSON with:
      n_hvg   - number of HVG genes selected
      sample  - first 20 gene names
    """
    try:
        import importlib.util, sys
        from pathlib import Path as _P

        script = _P(__file__).parent.parent.parent / "data_process_v0.0.py"
        spec = importlib.util.spec_from_file_location("_sc", str(script))
        m = importlib.util.module_from_spec(spec)
        sys.modules["_sc"] = m
        spec.loader.exec_module(m)

        out = _P(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        hvg_genes = m.step_hvg(
            h5ad_path=_P(h5ad), out_dir=out,
            batch_key=batch_key, n_top_genes=n_top_genes,
            hvg_sample_total=hvg_sample_total, seed=seed,
            min_genes=min_genes, max_genes=max_genes,
            min_counts=min_counts, max_counts=max_counts,
            max_pct_mt=max_pct_mt, mt_prefix=mt_prefix,
            qc_chunk_size=qc_chunk_size,
        )
        STORE["hvg_genes"] = hvg_genes
        return json.dumps({"status": "ok", "n_hvg": len(hvg_genes), "sample": hvg_genes[:20]})
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 2. TSV-based metadata  (L1000 / CLUE style: siginfo + cellinfo + compoundinfo)
# ---------------------------------------------------------------------------

@tool
def load_metadata_tsv(
    conditional:  str,
    cellinfo:     str,
    compoundinfo: str,
) -> str:
    """
    Load L1000/CLUE metadata from TSV files.

    Reads siginfo (conditional table), filters qc_pass==1, merges cellinfo
    (tissue, disease, sex, age) and compoundinfo (drug name fallback).
    Standardizes 15 metadata columns matching the single-cell pipeline.

    Stores result in STORE['tsv_metadata'] (DataFrame indexed by sample_id).

    Returns JSON with:
      rows    - number of QC-passed signatures
      columns - column names
    """
    try:
        import importlib.util, sys
        from pathlib import Path as _P

        script = _P(__file__).parent.parent.parent / "data_process_L1000_v0.1.py"
        spec = importlib.util.spec_from_file_location("_l1000", str(script))
        m = importlib.util.module_from_spec(spec)
        sys.modules["_l1000"] = m
        spec.loader.exec_module(m)

        cond_keep, _ = m.step_load_metadata(
            conditional_path=_P(conditional),
            cellinfo_path=_P(cellinfo),
            compoundinfo_path=_P(compoundinfo),
        )
        STORE["tsv_metadata"] = cond_keep
        return json.dumps({
            "status":  "ok",
            "rows":    len(cond_keep),
            "columns": cond_keep.columns.tolist(),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 3. HuggingFace Arrow dataset  (Tahoe-100M style: obs_metadata + cell_line_metadata)
# ---------------------------------------------------------------------------

@tool
def load_metadata_hf(
    dataset_dir:  str,
    subset_split: str = "train",
    qc_col:       str = "pass_filter",
    qc_val:       str = "full",
) -> str:
    """
    Load HuggingFace Arrow dataset metadata with QC filtering.

    Reads obs_metadata (filters qc_col == qc_val) and merges cell_line_metadata
    (cell_name → Organ/tissue).  Standardizes 15 metadata columns indexed
    by BARCODE_SUB_LIB_ID.

    Stores result in STORE['hf_metadata'] (DataFrame indexed by BARCODE_SUB_LIB_ID).

    Returns JSON with:
      rows    - number of QC-passed observations
      columns - column names
    """
    try:
        import importlib.util, sys
        from pathlib import Path as _P

        script = _P(__file__).parent.parent.parent / "data_process_Tahoe.py"
        spec = importlib.util.spec_from_file_location("_tahoe", str(script))
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tahoe"] = m
        spec.loader.exec_module(m)

        obs_keep = m.step_load_metadata(
            dataset_dir=_P(dataset_dir),
            subset_split=subset_split,
            qc_col=qc_col,
            qc_val=qc_val,
        )
        STORE["hf_metadata"] = obs_keep
        return json.dumps({
            "status":  "ok",
            "rows":    len(obs_keep),
            "columns": obs_keep.columns.tolist(),
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
