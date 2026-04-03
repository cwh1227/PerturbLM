# small_graph_subgraph

Subgraph extraction package for the `smaller_graph` knowledge graph (KG).  
Extracts a locally focused, biologically balanced subgraph around a seed entity (gene, drug, protein, pathway, or GO term) and formats it as structured text for downstream use.

---

## Overview

```
seed (name or ID)
      │
      ▼
  resolve_seed          ← fuzzy name → node ID lookup
      │
      ▼
  extract_subgraph      ← layered BFS with per-type budgets (from bmg_subgraph)
      │
      ▼
  format_raw_subgraph   ← structured text rendering (from bmg_subgraph)
      │
      ▼
  lookup embeddings     ← (optional) attach pre-trained KG node embeddings
      │
      ▼
  .jsonl output / stdout
```

---

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Path constants and default edge types |
| `indexing.py` | Build entity and adjacency indices (with pickle cache) |
| `resolver.py` | Resolve seed name → node ID with ranked fuzzy matching |
| `embedding.py` | Load pre-trained KG embeddings and look up subgraph node vectors |
| `pipeline.py` | Orchestrate single and batch extraction |
| `cli.py` | Command-line entry point |

---

## Data Dependencies

All paths are resolved relative to the **project root** (the directory you run from).

### KG Embeddings (for `--embed`)

Two pre-trained sources are tried in order:

| Priority | Source | Entity ID file | Embedding file | Shape |
|----------|--------|---------------|----------------|-------|
| 1 (preferred) | HeteroGraphSAGE 4-layer | `smaller_graph/training/entity2id.json` | `smaller_graph/training/heterographsage/4layer_entity_embeddings_all.pt` | [62507, 256] |
| 2 (fallback) | RotatE | `smaller_graph/training/rotate/entity_to_id.json` | `smaller_graph/training/rotate/entity_embeddings.pt` | [63715, 256] |

Both are `float32`, 256-dim. Nodes not present in the index are silently skipped  
(coverage is reported in `embedding_coverage`).

### Entity index (one required)
| Path | Notes |
|------|-------|
| `smaller_graph/training/entity2id.json` | preferred |
| `smaller_graph/training/rotate/entity_to_id.json` | fallback |
| `KG_data/output/entity2id.json` | fallback |

Format: `{ "gene:TP53": 0, "drug:imatinib": 1, … }` or a list of entity IDs.

### KG edges (one required)
| Path | Format |
|------|--------|
| `smaller_graph/training/kg_edges.parquet` | preferred — columns: `source`, `target`, `edge_type` |
| `smaller_graph/kg_edges.parquet` | fallback |
| `KG_data/output/kg_edges.parquet` | fallback |
| `smaller_graph/training/kg_triples.tsv` | TSV fallback — columns: `source  edge_type  target` (no header) |

### Gene descriptions (optional)
`preprocessing/hgnc_complete_set.txt` — HGNC TSV with `symbol` and `name` columns.  
Used to attach short descriptions to gene nodes. Extraction works without it.

---

## Usage

### Python API

```python
from small_graph_subgraph import run_pipeline, run_batch

# Single seed — text only
result = run_pipeline(seed="TP53", hops=2, max_nodes=60)
print(result["structured_text"])

# Single seed — with node embeddings
result = run_pipeline(seed="TP53", hops=2, max_nodes=60, embed=True)
node_embs = result["node_embeddings"]   # {node_id: [float x 256]}
print(result["embedding_source"])       # "heterographsage_4layer" or "rotate"
print(result["embedding_coverage"])     # {"found": 38, "missing": 2}

# Batch with embeddings (index loaded once, reused across all seeds)
run_batch(
    input_file="seeds.txt",   # one seed per line, # lines are comments
    output_file="out.jsonl",
    hops=2,
    max_nodes=60,
    embed=True,
)
```

### CLI

```bash
# Single seed
python -m small_graph_subgraph.cli --seed TP53

# With node embeddings
python -m small_graph_subgraph.cli --seed TP53 --embed --output results.jsonl

# With options
python -m small_graph_subgraph.cli \
    --seed imatinib \
    --hops 3 \
    --max_nodes 80 \
    --edge_types drug_targets,protein_interacts \
    --embed \
    --output results.jsonl

# Batch file with embeddings
python -m small_graph_subgraph.cli \
    --batch seeds.txt \
    --embed \
    --output out.jsonl

# Disable bidirectional traversal
python -m small_graph_subgraph.cli --seed EGFR --no_bidirectional

# Force rebuild cached indices
python -m small_graph_subgraph.cli --seed BRCA1 --rebuild_cache
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | — | Single seed name or node ID (mutually exclusive with `--batch`) |
| `--batch` | — | Text file with one seed per line |
| `--hops` | `2` | BFS traversal depth |
| `--max_nodes` | `60` | Maximum nodes in the extracted subgraph |
| `--edge_types` | all defaults | Comma-separated edge types to include |
| `--output` | stdout only | JSONL file path for saving results |
| `--embed` | `False` | Attach pre-trained KG node embeddings to output |
| `--rebuild_cache` | `False` | Delete and rebuild pickled index cache |
| `--no_bidirectional` | `False` | Disable automatic reverse edge traversal |

---

## Default Edge Types

```
drug_targets          gene_encodes_protein    gene_in_pathway
has_go_bp             has_go_cc               has_go_mf
in_complex_with       ligand_receptor         protein_interacts
tf_regulates
```

Reverse edges are added automatically (as `rev_<edge_type>`) when `bidirectional=True`.

---

## Output Format

Each result is a dict (or JSONL line in batch mode):

```json
{
  "seed": "TP53",
  "seed_name": "TP53",
  "seed_type": "Gene",
  "seed_ids": ["gene:TP53"],
  "hops": 2,
  "edge_types": ["drug_targets", "protein_interacts", "..."],
  "node_count": 42,
  "edge_count": 87,
  "structured_text": "...",
  "raw_nodes": {
    "gene:TP53": { "name": "TP53", "type": "Gene", "desc": "..." },
    "..."
  },
  "raw_edges": [
    { "src": "gene:TP53", "dst": "protein:P04637", "type": "gene_encodes_protein" },
    "..."
  ],

  // only present when embed=True
  "node_embeddings": {
    "gene:TP53":     [0.12, -0.34, "... 256 floats"],
    "drug:imatinib": [0.05,  0.91, "... 256 floats"],
    "..."
  },
  "embedding_source":   "heterographsage_4layer",
  "embedding_dim":      256,
  "embedding_coverage": { "found": 40, "missing": 2 }
}
```

---

## Index Cache

Parsed indices are cached as pickle files under `smaller_graph/.cache/` to speed up repeated runs.  
Cache is invalidated automatically when the version constant in `indexing.py` is bumped.  
Use `--rebuild_cache` to force a fresh rebuild.

---

## Dependencies

- `bmg_subgraph` (internal package — must be on `PYTHONPATH`)
- `pandas`, `pyarrow` (for parquet edge loading)
