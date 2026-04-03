"""Pipeline orchestration for smaller_graph subgraph extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from bmg_subgraph.extractor import extract_subgraph
    from bmg_subgraph.generator import format_raw_subgraph
    from bmg_subgraph.logging_utils import configure_logging
except ImportError as _e:
    raise ImportError(
        "small_graph_subgraph requires the 'bmg_subgraph' package to be installed. "
        f"Original error: {_e}"
    ) from _e

from .config import DEFAULT_EDGE_TYPES
from .embedding import EmbeddingIndex, load_embedding_index
from .indexing import build_entity_index, build_graph_index
from .resolver import resolve_seed


def _effective_allowed(allowed: set[str], bidirectional: bool) -> set[str]:
    if not bidirectional:
        return allowed
    return set(allowed) | {f"rev_{edge_type}" for edge_type in allowed}


def _run_single(
    seed: str,
    hops: int,
    max_nodes: int,
    allowed: set[str],
    traversal_types: set[str],
    id_to_name: dict,
    id_to_type: dict,
    name_to_ids: dict,
    id_to_desc: dict,
    adj: dict,
    output_file: Optional[str],
    log,
    emb_index: Optional[EmbeddingIndex] = None,
) -> dict:
    """Inner function that operates on pre-built indices."""
    seed_ids, seed_name, seed_type = resolve_seed(seed, name_to_ids, id_to_name, id_to_type)
    log.info("Seed resolved: %s [%s] -> %s", seed_name, seed_type, seed_ids)

    nodes, edges = extract_subgraph(
        seed_ids=seed_ids,
        adj=adj,
        id_to_name=id_to_name,
        id_to_type=id_to_type,
        id_to_desc=id_to_desc,
        max_hops=hops,
        max_nodes=max_nodes,
        allowed_edge_types=traversal_types,
    )
    log.info("Subgraph: %s nodes, %s edges", len(nodes), len(edges))

    structured_text = format_raw_subgraph(nodes, edges)
    result = {
        "seed": seed,
        "seed_name": seed_name,
        "seed_type": seed_type,
        "seed_ids": seed_ids,
        "hops": hops,
        "edge_types": sorted(allowed),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "structured_text": structured_text,
        "raw_nodes": dict(nodes),
        "raw_edges": edges,
    }

    if emb_index is not None:
        node_embeddings = emb_index.lookup(list(nodes.keys()))
        missing = len(nodes) - len(node_embeddings)
        result["node_embeddings"] = node_embeddings
        result["embedding_source"] = emb_index.source
        result["embedding_dim"] = emb_index.dim
        result["embedding_coverage"] = {
            "found": len(node_embeddings),
            "missing": missing,
        }
        log.info(
            "Embeddings: %d/%d nodes found (source: %s)",
            len(node_embeddings), len(nodes), emb_index.source,
        )

    print("\n" + "=" * 60)
    print(structured_text)
    print("=" * 60 + "\n")

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.info("Saved to %s", out_path)

    return result


def run_pipeline(
    seed: str,
    hops: int = 2,
    max_nodes: int = 60,
    edge_types: Optional[list[str]] = None,
    output_file: Optional[str] = None,
    bidirectional: bool = True,
    embed: bool = False,
    log=None,
) -> dict:
    """Extract a single smaller_graph subgraph.

    embed — when True, attach pre-trained KG node embeddings to the result
            under the key ``node_embeddings`` ({node_id: [float, ...]}).
            Tries HeteroGraphSAGE 4-layer first, falls back to RotatE.
    """
    log = log or configure_logging()
    allowed = set(edge_types) if edge_types else set(DEFAULT_EDGE_TYPES)
    traversal_types = _effective_allowed(allowed, bidirectional)

    id_to_name, id_to_type, name_to_ids, id_to_desc = build_entity_index(log)
    adj = build_graph_index(log, allowed_edge_types=allowed, bidirectional=bidirectional)
    emb_index = load_embedding_index(log) if embed else None

    return _run_single(
        seed=seed,
        hops=hops,
        max_nodes=max_nodes,
        allowed=allowed,
        traversal_types=traversal_types,
        id_to_name=id_to_name,
        id_to_type=id_to_type,
        name_to_ids=name_to_ids,
        id_to_desc=id_to_desc,
        adj=adj,
        output_file=output_file,
        log=log,
        emb_index=emb_index,
    )


def run_batch(
    input_file: str,
    hops: int = 2,
    max_nodes: int = 60,
    edge_types: Optional[list[str]] = None,
    output_file: str = "small_graph_subgraphs.jsonl",
    bidirectional: bool = True,
    embed: bool = False,
    log=None,
) -> None:
    """Extract smaller_graph subgraphs for a batch of seeds.

    embed — when True, attach pre-trained KG node embeddings to every result.
            The embedding index is loaded once and reused across all seeds.
    """
    log = log or configure_logging()
    allowed = set(edge_types) if edge_types else set(DEFAULT_EDGE_TYPES)
    traversal_types = _effective_allowed(allowed, bidirectional)

    # Build indices once for all seeds
    id_to_name, id_to_type, name_to_ids, id_to_desc = build_entity_index(log)
    adj = build_graph_index(log, allowed_edge_types=allowed, bidirectional=bidirectional)
    emb_index = load_embedding_index(log) if embed else None

    seeds = Path(input_file).read_text(encoding="utf-8").splitlines()
    seeds = [s.strip() for s in seeds if s.strip() and not s.startswith("#")]
    log.info("Batch: %s seeds -> %s", len(seeds), output_file)

    for index, seed in enumerate(seeds, start=1):
        log.info("[%s/%s] Processing: %s", index, len(seeds), seed)
        try:
            _run_single(
                seed=seed,
                hops=hops,
                max_nodes=max_nodes,
                allowed=allowed,
                traversal_types=traversal_types,
                id_to_name=id_to_name,
                id_to_type=id_to_type,
                name_to_ids=name_to_ids,
                id_to_desc=id_to_desc,
                adj=adj,
                output_file=output_file,
                log=log,
                emb_index=emb_index,
            )
        except Exception as exc:
            log.error("Failed for '%s': %s", seed, exc)
