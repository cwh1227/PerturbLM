"""
LangGraph ReAct graph for ATAC key-data extraction.

Architecture:
  START → agent ──(tool call)──► tools ──► agent
             └──(no tool call)──► END

Goal:
- From ATAC peak-to-gene records, output one score per gene.
- Core score definition follows user requirement:
    composite_score = mean peak-to-gene distance
  (default implementation uses mean absolute distance).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

try:
  _state_mod = importlib.import_module("Agent.state")
  _atac_mod = importlib.import_module("Agent.tools.atac")
except ModuleNotFoundError:
  _state_mod = importlib.import_module("state")
  try:
      _atac_mod = importlib.import_module("tools.atac")
  except ModuleNotFoundError:
      _atac_path = Path(__file__).parent / "tools" / "atac.py"
      _spec = importlib.util.spec_from_file_location("atac_module", _atac_path)
      if _spec is None or _spec.loader is None:
          raise RuntimeError(f"Failed to load module from {_atac_path}")
      _atac_mod = importlib.util.module_from_spec(_spec)
      _spec.loader.exec_module(_atac_mod)

AgentState = _state_mod.AgentState
compute_atac_gene_scores = _atac_mod.compute_atac_gene_scores
inspect_atac_table = _atac_mod.inspect_atac_table
inspect_atac_bundle = _atac_mod.inspect_atac_bundle
compute_atac_gene_scores_from_bundle = _atac_mod.compute_atac_gene_scores_from_bundle


def _create_llm(provider: str, model: str) -> Any:
    p = provider.lower()

    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=8192, temperature=0)

    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=8192, temperature=0)

    if p == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            max_tokens=8192,
            temperature=0,
        )

    if p == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            max_tokens=8192,
            temperature=0,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Choose from: anthropic, openai, qwen, deepseek"
    )


ATAC_TOOLS = [
    inspect_atac_table,
    compute_atac_gene_scores,
  inspect_atac_bundle,
  compute_atac_gene_scores_from_bundle,
]


SYSTEM_PROMPT_ATAC = """You are an ATAC-seq key-feature extraction agent.

Your job is to produce standardized ATAC gene-score outputs from either:
  A) a peak-to-gene table file, or
  B) a scATAC bundle directory (peak_bc/gene_scores/ChromVar/LSI_embedding/markerpeak_target).

The output format is always the same gene-score table files:
    gene                     str      [G]
    n_peaks                  int32    [G]
    mean_distance            float64  [G]
    median_distance          float64  [G]
    mean_abs_distance        float64  [G]
    composite_score          float64  [G]  — mean(|distance|) by default
    weighted_composite_score float64  [G]  — openness-weighted distance mean
    rank                     int32    [G]

WORKFLOW
────────
Reading and processing are separated.  Step 2 is input-specific;
step 3 output schema is unified.

1. INSPECT  — understand source format before doing anything:
   inspect_atac_table(path)         for flat file (.csv/.tsv/.txt/.parquet)
   inspect_atac_bundle(dataset_dir) for bundle directory input

2. COMPUTE GENE SCORES
   ┌───────────────────────────────┬────────────────────────────────────────────────────────────┐
   │ Input type                    │ Tool                                                       │
   ├───────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ Peak-to-gene flat table       │ compute_atac_gene_scores(path, out_dir, gene_col,         │
   │                               │ distance_col, openness_col, use_absolute_distance, ...)   │
   │ scATAC bundle directory       │ compute_atac_gene_scores_from_bundle(dataset_dir, out_dir,│
   │                               │ gene_col, distance_col, ... + QC thresholds)              │
   └───────────────────────────────┴────────────────────────────────────────────────────────────┘

   CRITICAL — decide from inspect output:
     gene_col      gene symbol column (e.g. nearestGene)
     distance_col  peak-to-gene distance column (e.g. distToTSS)
     openness_col  optional openness/intensity (flat table), if available

3. OUTPUT + SUMMARY  — same for all inputs:
   Writes to out_dir:
     atac_gene_scores.tsv
     atac_gene_scores.parquet
     atac_gene_scores_summary.json
   Summary must include n_genes, score_definition, top_genes_preview, output paths.

QC FILTER (bundle mode)
───────────────────────
For compute_atac_gene_scores_from_bundle, apply cell-level QC on peak_bc/obs.csv:
  TSSEnrichment  >= min_tss_enrichment
  FRIP           >= min_frip
  nFrags         >= min_fragments
  BlacklistRatio <= max_blacklist_ratio

Then compute per-peak openness from QC-passing cells using peak_bc/counts.mtx.gz,
and aggregate to gene-level scores via peak_bc/var.csv.

CONFIRMED DATASET PROFILE (Liscovitch-BrauerSanjana2021_K562_1)
────────────────────────────────────────────────────────────────
Bundle path example:
  C:/Users/hc858/OneDrive/Desktop/llm/sc_atac_seq_data/Liscovitch-BrauerSanjana2021_K562_1

Observed sub-datasets and shapes:
  peak_bc:
    counts.mtx.gz      shape [8723, 91181], nnz=3806533
    obs.csv            rows=8723, cols=28
    var.csv            rows=91181, cols=12
  gene_scores:
    counts.mtx.gz      shape [8723, 23127], nnz=11465966
    obs.csv            rows=8723, cols=28
    var.csv            rows=23127, cols=3
  ChromVar:
    counts.mtx.gz      shape [8723, 2174]  (array format)
    obs.csv            rows=8723, cols=28
    var.csv            rows=2174, cols=3
  LSI_embedding:
    counts.mtx.gz      shape [8723, 30]    (array format)
    obs.csv            rows=8723, cols=28
    var.csv            rows=30, cols=3
  markerpeak_target:
    counts.mtx.gz      shape [22, 876]     (array format)
    obs.csv            rows=22, cols=3
    var.csv            rows=876, cols=3

Confirmed columns — peak_bc/obs.csv (28):
  cell_barcode, BlacklistRatio, nDiFrags, nFrags, PromoterRatio,
  ReadsInBlacklist, ReadsInPromoter, ReadsInTSS, sample, TSSEnrichment,
  guide, sgCounts, sgTotal, sgSpec, perturbation, ReadsInPeaks, FRIP,
  disease, cancer, tissue_type, cell_line, organism, perturbation_type,
  nperts, percent_mito, percent_ribo, ncounts, ngenes

Confirmed columns — peak_bc/var.csv (12):
  peak_index, chromosome, start, end, distToGeneStart, nearestGene,
  peakType, distToTSS, nearestTSS, GC, ncounts, ncells

Recommended mapping for this dataset:
  gene_col      = nearestGene
  distance_col  = distToTSS
  openness_col  = ncounts  (or QC-recomputed openness from matrix)

RULES
─────
- Always inspect first — never guess columns or shapes.
- In bundle mode, run QC before final score aggregation.
- If required columns are missing, stop and report exact missing names.
- After each tool call, check returned JSON for "error" and stop if found.

EXTENSIBILITY
─────────────
For a new ATAC source format, add only parsing support in tools.atac;
the inspect → compute → summarize pattern stays unchanged.
"""


def build_graph_atac(
    model: str = "claude-sonnet-4-6",
    provider: str = "anthropic",
) -> Any:
    """Build and compile the ATAC LangGraph ReAct agent."""
    llm = _create_llm(provider, model)
    llm_with_tools = llm.bind_tools(ATAC_TOOLS)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ATAC_TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()
