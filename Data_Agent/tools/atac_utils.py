"""
Shared utilities for ATAC tools.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_GENE_CANDIDATES = [
    "nearestGene",
    "gene",
    "gene_symbol",
    "symbol",
    "nearest_gene",
    "target_gene",
    "gene_name",
    "hgnc_symbol",
]

_DISTANCE_CANDIDATES = [
    "distToTSS",
    "distance",
    "dist_to_tss",
    "tss_distance",
    "peak_to_gene_distance",
    "distance_to_gene",
    "dist",
]

_OPENNESS_CANDIDATES = [
    "ncounts",
    "score",
    "accessibility",
    "open_score",
    "signal",
    "peak_score",
    "counts",
]

# obs.csv column → standardized cell_meta column
QC_COL_MAP = {
    "TSSEnrichment": "tss_enrichment",
    "FRIP": "frip",
    "nFrags": "n_fragments",
    "BlacklistRatio": "blacklist_ratio",
}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, sep=None, engine="python")


def guess_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    return ""
