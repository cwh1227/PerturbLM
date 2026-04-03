"""KG embedding lookup for smaller_graph subgraph nodes.

Supports two pre-trained sources (tried in order):
  1. HeteroGraphSAGE 4-layer  — smaller_graph/training/entity2id.json
                                 + training/heterographsage/4layer_entity_embeddings_all.pt
  2. RotatE                   — smaller_graph/training/rotate/entity_to_id.json
                                 + training/rotate/entity_embeddings.pt

Both are 256-dim float32 tensors indexed by the same entity-ID format
(e.g. "gene:TP53", "drug:imatinib").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ROTATE_DIR, TRAINING_DIR

# ---------------------------------------------------------------------------
# Candidate paths
# ---------------------------------------------------------------------------

_GRAPHSAGE_ID_FILE  = TRAINING_DIR / "entity2id.json"
_GRAPHSAGE_EMB_FILE = TRAINING_DIR / "heterographsage" / "4layer_entity_embeddings_all.pt"

_ROTATE_ID_FILE  = ROTATE_DIR / "entity_to_id.json"
_ROTATE_EMB_FILE = ROTATE_DIR / "entity_embeddings.pt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_torch_as_numpy(path: Path) -> np.ndarray:
    """Load a .pt tensor without keeping torch in the public API."""
    import torch
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor.numpy().astype(np.float32)


def _try_load(id_file: Path, emb_file: Path, log) -> Optional[tuple[dict[str, int], np.ndarray]]:
    if not id_file.exists():
        log.debug("Embedding id file not found: %s", id_file)
        return None
    if not emb_file.exists():
        log.debug("Embedding matrix not found: %s", emb_file)
        return None
    with id_file.open("r", encoding="utf-8") as fh:
        entity_to_id: dict[str, int] = json.load(fh)
    matrix = _load_torch_as_numpy(emb_file)
    if matrix.shape[0] != len(entity_to_id):
        log.warning(
            "Embedding matrix rows (%d) != entity count (%d) in %s — skipping",
            matrix.shape[0], len(entity_to_id), emb_file,
        )
        return None
    log.info(
        "Loaded embeddings: %s  shape=%s",
        emb_file.name, matrix.shape,
    )
    return entity_to_id, matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Wraps a pre-trained KG embedding matrix for fast node lookup."""

    def __init__(self, entity_to_id: dict[str, int], matrix: np.ndarray, source: str) -> None:
        self._entity_to_id = entity_to_id
        self._matrix = matrix
        self.source = source
        self.dim = matrix.shape[1]
        self.n_entities = matrix.shape[0]

    def lookup(self, node_ids: list[str]) -> dict[str, list[float]]:
        """Return {node_id: embedding_vector} for each ID found in the index.

        IDs not present in the index are silently skipped.
        Vectors are returned as plain Python lists for JSON serialisation.
        """
        result: dict[str, list[float]] = {}
        for node_id in node_ids:
            idx = self._entity_to_id.get(node_id)
            if idx is not None:
                result[node_id] = self._matrix[idx].tolist()
        return result


def load_embedding_index(log) -> EmbeddingIndex:
    """Load the best available embedding index.

    Tries HeteroGraphSAGE 4-layer first, then RotatE.
    Raises FileNotFoundError if neither source is available.
    """
    pair = _try_load(_GRAPHSAGE_ID_FILE, _GRAPHSAGE_EMB_FILE, log)
    if pair is not None:
        return EmbeddingIndex(*pair, source="heterographsage_4layer")

    pair = _try_load(_ROTATE_ID_FILE, _ROTATE_EMB_FILE, log)
    if pair is not None:
        return EmbeddingIndex(*pair, source="rotate")

    raise FileNotFoundError(
        "No embedding files found. Checked:\n"
        f"  HeteroGraphSAGE: {_GRAPHSAGE_EMB_FILE}\n"
        f"  RotatE:          {_ROTATE_EMB_FILE}"
    )
