"""
Collect all tool functions into ALL_TOOLS for registration with the LangGraph agent.

RNA pipeline (7 steps):
  1. inspect_*           — show column names + sample values so the agent can decide mappings
  2. read_*              — write standardized intermediate parquet files to disk
                           (agent supplies column_map / qc_col / qc_val based on inspect output)
  3. normalize_celltypes — unify cellType1/2/majorCluster/subCluster values across datasets
                           (resolves abbreviations like NK→Natural Killer Cell, DC→Dendritic Cell
                            and separator/case differences; optional but recommended)
  4. qc_filter_cells     — filter cells by qc_pass flag + optional numeric thresholds
  5. select_hvg          — compute per-gene variance, mark top N as is_hvg=True
  6. map_to_hgnc         — map ensembl_id / gene_symbol → HGNC numeric ID, build vocab
  7. tokenize            — read parquet + original expression, write .npz shards

Steps 3–7 read only standardized intermediate/cell_meta.parquet + gene_meta.parquet
and are completely format-agnostic.

ATAC pipeline (Phase 1 — 4 steps):
  1. inspect_atac_*           — show format, columns, matrix shapes
  2. read_atac_*              — write cell_meta / peak_meta / peak_gene_link / data_source
  3. qc_filter_atac_cells     — update qc_pass in cell_meta.parquet (skipped for flat tables)
  4. aggregate_peaks_to_genes — compute gene-level scores, write atac_gene_scores.*
"""

from Agent.tools.inspect import (
    inspect_h5ad,
    inspect_gctx,
    inspect_hf_dataset,
    inspect_tsv,
)
from Agent.tools.read import (
    read_h5ad,
    read_gctx,
    read_hf_dataset,
)
from Agent.tools.qc import (
    compute_qc_metrics,
    normalize_celltypes,
    qc_filter_cells,
    select_hvg,
)
from Agent.tools.atac_inspect import (
    inspect_atac_table,
    inspect_atac_bundle,
)
from Agent.tools.atac_read import (
    read_atac_table,
    read_atac_bundle,
)
from Agent.tools.atac_qc import qc_filter_atac_cells
from Agent.tools.atac_aggregate import aggregate_peaks_to_genes

from Agent.tools.gene_vocab import map_to_hgnc
from Agent.tools.tokenize import tokenize


ALL_TOOLS = [
    # ── Step 1: Inspect — show columns + sample values for the agent to decide mappings
    inspect_h5ad,         # AnnData .h5ad — obs/var columns, n_cells, n_genes
    inspect_gctx,         # L1000 gctx — matrix shape, siginfo/geneinfo columns
    inspect_hf_dataset,   # HuggingFace Arrow — subset names, obs/gene columns
    inspect_tsv,          # any TSV — columns + sample rows
    # ── Step 2: Read — write cell_meta.parquet + gene_meta.parquet + data_source.json
    read_h5ad,            # AnnData .h5ad  (column_map, qc_col, qc_val)
    read_gctx,            # L1000 gctx + siginfo/cellinfo/compoundinfo/geneinfo TSVs
    read_hf_dataset,      # HuggingFace Arrow dataset (Tahoe style)
    # ── Step 3: QC + normalisation — reads/writes cell_meta.parquet
    compute_qc_metrics,   # compute n_genes/n_counts/pct_mt from expression matrix (when NaN)
    normalize_celltypes,  # unify cellType1/2/majorCluster/subCluster across datasets (abbrev + case)
    qc_filter_cells,      # remove qc_pass=False + optional min/max thresholds
    # ── Step 4: HVG selection — reads gene_meta.parquet + expression (format-agnostic)
    select_hvg,           # compute variance, mark top N genes as is_hvg=True
    # ── Step 5: HGNC mapping — reads/writes gene_meta.parquet (format-agnostic)
    map_to_hgnc,          # ensembl_id/gene_symbol → HGNC numeric ID, build gene_vocab.json
    # ── Step 6: Tokenize + shard — format auto-detected from data_source.json
    tokenize,             # write .npz shards (sample_id, gene_ids, bin_ids, lengths, …)
]

ALL_TOOLS_ATAC = [
    # ── Step 1: Inspect — understand format before reading
    inspect_atac_table,         # flat peak-to-gene table — columns, sample rows, guessed cols
    inspect_atac_bundle,        # scATAC bundle dir — subfolder shapes, recommended columns
    # ── Step 2: Read — write standardized intermediate files
    read_atac_table,            # flat table → peak_gene_link.parquet + data_source.json
    read_atac_bundle,           # bundle → cell_meta + peak_meta + peak_gene_link + data_source
    # ── Step 3: QC — filter cells (skipped automatically for flat table input)
    qc_filter_atac_cells,       # update qc_pass in cell_meta.parquet
    # ── Step 4: Aggregate — compute gene-level scores from intermediate files
    aggregate_peaks_to_genes,   # → atac_gene_scores.tsv / .parquet / _summary.json
]