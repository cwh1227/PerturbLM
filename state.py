"""
Shared agent state and intermediate file helpers.

All tools write to / read from a single intermediate directory on disk:
    intermediate/
        cell_meta.parquet    — all cells × standardized metadata + QC columns
        gene_meta.parquet    — all genes × gene info + HVG + HGNC columns
        data_source.json     — {format, path, ...} for expression data

Standardized cell_meta columns
───────────────────────────────
  sample_id  (index) str        unique cell / sample identifier
  qc_pass            bool       True = passes QC  ← always standardized at read time
  n_genes            int/NaN    detected gene count (from precomputed obs if available)
  n_counts           int/NaN    total counts
  pct_mt             float/NaN  % mitochondrial reads
  tissue             str        \
  region             str         |
  sampleType         str         |
  compartment        str         |
  cellType1          str         |  15 standard metadata columns
  cellType2          str         |
  majorCluster       str         |
  subCluster         str         |
  sex                str         |
  age                str         |
  perturbType        str         |
  disease            str         |
  drug               str         |
  dose               str         |
  time               str        /

Standardized gene_meta columns
───────────────────────────────
  orig_id       str   original identifier (gene name / entrez_id / token_id string)
  gene_symbol   str   gene symbol
  ensembl_id    str   Ensembl ID (may be empty)
  gctx_index    int   position in gctx ROW axis (gctx datasets only, else -1)
  orig_token_id int   original HF token_id integer (HF datasets only, else -1)
  is_hvg        bool  set by select_hvg  (True for all genes initially)
  hgnc_num      str   HGNC numeric ID string (set by map_to_hgnc, "" if unmapped)
  hgnc_token    int   token ID in gene_vocab (set by map_to_hgnc, 0 if unmapped)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Legacy scratch store for older helper tools not used by the main graph.
STORE: Dict[str, Any] = {}

def get_intermediate_dir(intermediate_dir: str = "") -> Path:
    if not intermediate_dir:
        raise RuntimeError("intermediate_dir not set — call a read_* tool first")
    return Path(intermediate_dir)


def cell_meta_path(intermediate_dir: str = "") -> Path:
    return get_intermediate_dir(intermediate_dir) / "cell_meta.parquet"


def gene_meta_path(intermediate_dir: str = "") -> Path:
    return get_intermediate_dir(intermediate_dir) / "gene_meta.parquet"


def data_source_path(intermediate_dir: str = "") -> Path:
    return get_intermediate_dir(intermediate_dir) / "data_source.json"


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    out_dir: str
    intermediate_dir: str
