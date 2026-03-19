"""
LangGraph ReAct agent graph.

Architecture:
  START → agent ──(tool call)──► tools ──► agent
             └──(no tool call)──► END

Supported LLM providers:
  anthropic  — Claude (default); requires ANTHROPIC_API_KEY
  openai     — GPT-4o etc.;     requires OPENAI_API_KEY
  qwen       — Qwen (Aliyun);   requires DASHSCOPE_API_KEY
  deepseek   — DeepSeek;        requires DEEPSEEK_API_KEY
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from Agent.state import AgentState
from Agent.tools import ALL_TOOLS, ALL_TOOLS_ATAC


SYSTEM_PROMPT = """You are a genomics data-processing agent.

Your job is to produce standardized gene-token shards from any single-cell or
bulk omics dataset.  The output format is always the same compressed .npz shard:
    gene_ids  int32  [B, K]  — HGNC-numeric-ID-based gene tokens
    bin_ids   int16  [B, K]  — rank-based expression bins
    lengths   int16  [B]     — real sequence lengths
    text_data U      [B, C]  — metadata (tissue, drug, dose, …)
    domain_id int32  [B]     — batch/domain label
    sample_id U      [B]     — cell/sample identifier

WORKFLOW
────────
Reading and processing are separated.  Step 2 is format-specific;
steps 3–6 are completely format-agnostic (same tool, same parquet files).

1. INSPECT  — understand file format before doing anything:
   inspect_h5ad / inspect_gctx / inspect_hf_dataset / inspect_tsv

2. READ  — convert to standardized parquet files (no QC, no filtering):
   ┌────────────────────────┬──────────────────────────────────────────┐
   │ Format                 │ Tool                                     │
   ├────────────────────────┼──────────────────────────────────────────┤
   │ AnnData (.h5ad)        │ read_h5ad(path, column_map, qc_col, …)   │
   │ gctx + TSV             │ read_gctx(gctx, siginfo, …,              │
   │                        │          column_map, qc_col, qc_val, …)  │
   │ HuggingFace Arrow      │ read_hf_dataset(dataset_dir,             │
   │                        │          column_map, qc_col, qc_val, …)  │
   └────────────────────────┴──────────────────────────────────────────┘

   CRITICAL — before calling read_*, decide from the inspect output:
     column_map  JSON dict: standard_col → actual source column name found.
                 Example: {"tissue": "tissue_general", "cellType1": "cell_type",
                           "drug": "pert_name", "dose": "pert_dose",
                           "perturbType": "pert_type", "sex": "donor_sex"}
                 Omit any standard column not found; it will be empty string.
     qc_col / qc_val  column + value marking a passing cell/sample.
                 Example: qc_col="qc_pass", qc_val="1"  (L1000)
                          qc_col="pass_filter", qc_val="full"  (Tahoe/HF)
                 Leave empty for h5ad if no pre-computed QC flag exists.

   All format differences are resolved HERE. Writes to intermediate/:
     cell_meta.parquet  — all cells, standardized columns:
       qc_pass (bool)    ← standardized from dataset's own QC flag
       n_genes / n_counts / pct_mt  ← standardized QC metric columns
       tissue / region / sampleType / ... (15 standard metadata cols)
     gene_meta.parquet  — all genes: orig_id, gene_symbol, ensembl_id,
                          gctx_index, orig_token_id, is_hvg, hgnc_num, hgnc_token
     data_source.json   — {format, path, ...} for expression data

3. QC FILTER  — same tools for ALL formats (reads/writes cell_meta.parquet):

   Decision based on whether a QC flag existed in the original data:

   A. Dataset HAS a pre-computed QC flag (qc_col was provided to read_*):
      → qc_filter_cells(intermediate_dir=...)
        Removes cells where qc_pass=False. No numeric thresholds needed.

   B. Dataset has NO QC flag (qc_col was left empty → qc_pass=True for all):
      Applies to single-cell h5ad files without a pre-computed QC column.
      → compute_qc_metrics(mt_prefix="MT-", intermediate_dir=...)
        Computes n_genes, n_counts, pct_mt from expression matrix.
        mt_prefix: "MT-" for human, "mt-" for mouse.
      → qc_filter_cells(min_genes=500, max_genes=8000, min_counts=1000,
                        max_counts=100000, max_pct_mt=20.0, intermediate_dir=...)
        Filters by numeric thresholds on the newly computed metrics.

4. HVG SELECTION  — same tool for ALL formats (reads/writes gene_meta.parquet):
   select_hvg(n_top_genes=3000)
   Computes per-gene variance (samples expression data internally by format),
   marks top n_top_genes as is_hvg=True.
   For bulk L1000 (978 fixed landmark genes) this step is optional.

5. HGNC MAPPING + VOCABULARY  — same tool for ALL formats:
   map_to_hgnc(hgnc_path, out_dir)
   Maps ensembl_id → HGNC (primary), gene_symbol → HGNC (fallback).
   Only is_hvg=True genes with a valid HGNC ID enter the vocabulary.
   Outputs gene_vocab.json, conn_id_to_hgnc.json, gene_mapping_table.tsv.

6. TOKENIZE + SHARD  — single tool, format auto-detected from data_source.json:
   tokenize(out_dir, topk=1024, n_bins=32, ...)
   Reads cell_meta.parquet + gene_meta.parquet + original expression data.
   Output is always the same .npz shard schema regardless of input format.

RULES
─────
- Always inspect files first — never guess the format.
- The HGNC reference file (hgnc_complete_set.txt) is required for step 5.
  If not provided, ask the user for its path.
- After each tool call, check the returned JSON for "error" and stop if found.
- After all steps succeed, summarize n_shards, vocab_size, and out_dir.

EXTENSIBILITY
─────────────
For a new data format: add only a read_* tool that writes the standard
cell_meta.parquet + gene_meta.parquet schema.  Steps 3–6 work unchanged.
"""

SYSTEM_PROMPT_ATAC = """You are an ATAC-seq key-feature extraction agent.

Your job is to produce standardized ATAC gene-score outputs from either:
  A) a peak-to-gene table file (.csv/.tsv/.txt/.parquet), or
  B) a scATAC bundle directory (contains peak_bc/ with obs.csv, var.csv, counts.mtx.gz).

Final output (always the same schema):
    gene                     str      [G]
    n_peaks                  int32    [G]
    mean_distance            float64  [G]
    median_distance          float64  [G]
    mean_abs_distance        float64  [G]
    composite_score          float64  [G]  — mean(|distance|) by default
    weighted_composite_score float64  [G]  — openness-weighted distance mean (when openness available)
    rank                     int32    [G]

WORKFLOW
────────
Reading and aggregation are separated.  Step 2 is format-specific;
steps 3–4 are completely format-agnostic (same tools, same intermediate files).

Intermediate files written to intermediate_dir:
  cell_meta.parquet      — per-cell QC metrics, qc_pass flag  (bundle mode only)
  peak_meta.parquet      — per-peak features from var.csv      (bundle mode only)
  peak_gene_link.parquet — peak-to-gene links: peak_id, gene_symbol,
                           distance_to_tss, openness, link_type, is_valid_link
  data_source.json       — source format, paths, column names used

1. INSPECT  — understand format before doing anything:
   inspect_atac_table(path)         → flat file
   inspect_atac_bundle(dataset_dir) → bundle directory

2. READ  — parse raw input, write standardized intermediate files:
   ┌───────────────────────────────┬──────────────────────────────────────────────────────────┐
   │ Input type                    │ Tool                                                     │
   ├───────────────────────────────┼──────────────────────────────────────────────────────────┤
   │ Flat peak-to-gene table       │ read_atac_table(path, intermediate_dir,                  │
   │                               │   gene_col, distance_col, openness_col)                  │
   │                               │ Writes: peak_gene_link.parquet + data_source.json        │
   │                               │ No cell_meta (flat tables have no cell-level info)       │
   ├───────────────────────────────┼──────────────────────────────────────────────────────────┤
   │ scATAC bundle directory       │ read_atac_bundle(dataset_dir, intermediate_dir,          │
   │                               │   gene_col, distance_col)                                │
   │                               │ Writes: cell_meta + peak_meta + peak_gene_link +        │
   │                               │         data_source.json                                 │
   │                               │ Matrix NOT loaded here — deferred to step 4              │
   └───────────────────────────────┴──────────────────────────────────────────────────────────┘

   CRITICAL — decide from inspect output:
     gene_col      gene symbol column (e.g. nearestGene)
     distance_col  peak-to-TSS distance column (e.g. distToTSS)
     openness_col  optional openness/intensity (flat table only)

3. QC FILTER  — same tool for ALL formats:
   qc_filter_atac_cells(intermediate_dir,
     min_tss_enrichment=4.0, min_frip=0.15,
     min_fragments=3000, max_blacklist_ratio=0.05)

   Reads/writes cell_meta.parquet; updates qc_pass column.
   AUTOMATICALLY SKIPPED for flat table input (data_source.json has_cell_meta=false).
   Standardized QC columns: tss_enrichment, frip, n_fragments, blacklist_ratio.
   Absent columns are silently skipped.

4. AGGREGATE  — same tool for ALL formats:
   aggregate_peaks_to_genes(intermediate_dir, out_dir,
     use_absolute_distance=True, min_peaks_per_gene=1)

   Reads peak_gene_link.parquet + data_source.json (+ cell_meta + matrix for bundle).
   Bundle mode: loads counts.mtx.gz, applies QC cell mask, computes per-peak openness.
   Flat table:  uses openness column already in peak_gene_link.parquet.

   Writes to out_dir:
     atac_gene_scores.tsv
     atac_gene_scores.parquet
     atac_gene_scores_summary.json

RULES
─────
- Always inspect first — never guess columns or shapes.
- After each tool call, check returned JSON for "error" and stop if found.
- If required columns are missing, stop and report exact missing names.
- After all steps succeed, summarize n_genes, openness_source, and out_dir.

EXTENSIBILITY
─────────────
For a new ATAC source format: add only a read_atac_* tool that writes the standard
cell_meta + peak_meta + peak_gene_link + data_source schema.
Steps 3–4 (qc_filter_atac_cells, aggregate_peaks_to_genes) work unchanged.
"""

# ---------------------------------------------------------------------------
# LLM factory — selects provider based on the `provider` argument
# ---------------------------------------------------------------------------

def _create_llm(provider: str, model: str) -> Any:
    """
    Instantiate a LangChain chat model for the given provider.

    Supported providers and required environment variables:
      anthropic  — ANTHROPIC_API_KEY   (default)
      openai     — OPENAI_API_KEY
      qwen       — DASHSCOPE_API_KEY   (OpenAI-compatible endpoint)
      deepseek   — DEEPSEEK_API_KEY    (OpenAI-compatible endpoint)
    """
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


def build_graph_atac(model: str = "claude-sonnet-4-6", provider: str = "anthropic") -> Any:
    """Build and compile the ATAC LangGraph ReAct agent."""
    llm = _create_llm(provider, model)
    llm_with_tools = llm.bind_tools(ALL_TOOLS_ATAC)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ALL_TOOLS_ATAC))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()


def build_graph(model: str = "claude-sonnet-4-6", provider: str = "anthropic") -> Any:
    """Build and compile the LangGraph ReAct agent."""
    llm = _create_llm(provider, model)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()
