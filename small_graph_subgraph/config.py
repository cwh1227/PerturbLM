"""Static configuration for smaller_graph subgraph extraction."""

from pathlib import Path

SMALL_GRAPH_ROOT = Path("smaller_graph")
TRAINING_DIR = SMALL_GRAPH_ROOT / "training"
ROTATE_DIR = TRAINING_DIR / "rotate"
CACHE_DIR = SMALL_GRAPH_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HGNC_FILE = Path("preprocessing") / "hgnc_complete_set.txt"

ENTITY_INDEX_CANDIDATES = [
    TRAINING_DIR / "entity2id.json",
    ROTATE_DIR / "entity_to_id.json",
    Path("KG_data") / "output" / "entity2id.json",
]

RELATION_INDEX_CANDIDATES = [
    ROTATE_DIR / "relation_to_id.json",
]

KG_EDGE_PARQUET_CANDIDATES = [
    TRAINING_DIR / "kg_edges.parquet",
    SMALL_GRAPH_ROOT / "kg_edges.parquet",
    SMALL_GRAPH_ROOT / "output" / "kg_edges.parquet",
    Path("KG_data") / "output" / "kg_edges.parquet",
]

KG_TRIPLES_CANDIDATES = [
    TRAINING_DIR / "kg_triples.tsv",
    SMALL_GRAPH_ROOT / "kg_triples.tsv",
    SMALL_GRAPH_ROOT / "output" / "kg_triples.tsv",
    Path("KG_data") / "output" / "kg_triples.tsv",
]

DEFAULT_EDGE_TYPES = [
    "drug_targets",
    "gene_encodes_protein",
    "gene_in_pathway",
    "has_go_bp",
    "has_go_cc",
    "has_go_mf",
    "in_complex_with",
    "ligand_receptor",
    "protein_interacts",
    "tf_regulates",
]

TYPE_PREFIX_TO_NAME = {
    "drug": "Drug",
    "gene": "Gene",
    "protein": "Protein",
    "go": "GO",
    "pathway": "Pathway",
}

TYPE_PRIORITY = [
    "Drug",
    "Gene",
    "Protein",
    "Pathway",
    "GO",
]
