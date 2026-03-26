"""Pipeline orchestration for single and batch extraction."""

import json
from pathlib import Path
from typing import Optional

from .config import BIDIRECTIONAL_EDGE_TYPES, DEFAULT_EDGE_TYPES
from .extractor import extract_subgraph
from .generator import LLMConfig, create_llm_client, format_raw_subgraph, generate_subgraph_text
from .indexing import build_entity_index, build_graph_index
from .resolver import resolve_seed


def _effective_allowed(allowed: set[str], bidirectional: bool) -> set[str]:
    """Expand *allowed* to include synthesised reverse edge types.

    When bidirectional=True, edges like Drug-Protein are stored in the graph
    index as both Drug-Protein and Protein-Drug.  The extractor must also
    be told that Protein-Drug is a valid edge type, otherwise it filters them out.
    """
    if not bidirectional:
        return allowed
    extra: set[str] = set()
    for etype in list(allowed):
        if etype in BIDIRECTIONAL_EDGE_TYPES:
            parts = etype.split("-", 1)
            if len(parts) == 2:
                extra.add(f"{parts[1]}-{parts[0]}")
    return allowed | extra


def run_pipeline(
    log,
    seed: str,
    hops: int = 2,
    max_nodes: int = 60,
    edge_types: Optional[list[str]] = None,
    llm_config: Optional[LLMConfig] = None,
    output_file: Optional[str] = None,
    dry_run: bool = False,
    bidirectional: bool = True,
) -> dict:
    allowed = set(edge_types) if edge_types else set(DEFAULT_EDGE_TYPES)
    llm_config = llm_config or LLMConfig()
    id_to_name, id_to_type, name_to_ids, id_to_desc = build_entity_index(log)
    adj = build_graph_index(log, allowed_edge_types=allowed, bidirectional=bidirectional)
    traversal_types = _effective_allowed(allowed, bidirectional)

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

    if dry_run:
        structured_text = format_raw_subgraph(nodes, edges)
    else:
        client = create_llm_client(llm_config)
        structured_text = generate_subgraph_text(
            seed_name=seed_name,
            seed_type=seed_type,
            nodes=nodes,
            edges=edges,
            client=client,
            config=llm_config,
        )

    result = {
        "seed": seed,
        "seed_name": seed_name,
        "seed_type": seed_type,
        "seed_ids": seed_ids,
        "hops": hops,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.model,
        "structured_text": structured_text,
        "raw_nodes": dict(nodes),
        "raw_edges": edges,
    }

    print("\n" + "=" * 60)
    print(structured_text)
    print("=" * 60 + "\n")

    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.info("Saved to %s", output_file)

    return result


def run_batch(
    log,
    input_file: str,
    hops: int = 2,
    max_nodes: int = 60,
    edge_types: Optional[list[str]] = None,
    llm_config: Optional[LLMConfig] = None,
    output_file: str = "training_data.jsonl",
    dry_run: bool = False,
    bidirectional: bool = True,
) -> None:
    """Process a list of seed entities from a text file."""
    allowed = set(edge_types) if edge_types else set(DEFAULT_EDGE_TYPES)
    llm_config = llm_config or LLMConfig()
    id_to_name, id_to_type, name_to_ids, id_to_desc = build_entity_index(log)
    adj = build_graph_index(log, allowed_edge_types=allowed, bidirectional=bidirectional)
    traversal_types = _effective_allowed(allowed, bidirectional)
    client = None if dry_run else create_llm_client(llm_config)

    seeds = Path(input_file).read_text(encoding="utf-8").splitlines()
    seeds = [seed.strip() for seed in seeds if seed.strip() and not seed.startswith("#")]
    log.info("Batch: %s seeds -> %s", len(seeds), output_file)

    for index, seed in enumerate(seeds, start=1):
        log.info("[%s/%s] Processing: %s", index, len(seeds), seed)
        try:
            seed_ids, seed_name, seed_type = resolve_seed(
                seed,
                name_to_ids,
                id_to_name,
                id_to_type,
            )
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

            if dry_run:
                structured_text = format_raw_subgraph(nodes, edges)
            else:
                structured_text = generate_subgraph_text(
                    seed_name=seed_name,
                    seed_type=seed_type,
                    nodes=nodes,
                    edges=edges,
                    client=client,
                    config=llm_config,
                )

            result = {
                "seed": seed,
                "seed_name": seed_name,
                "seed_type": seed_type,
                "hops": hops,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "llm_provider": llm_config.provider,
                "llm_model": llm_config.model,
                "structured_text": structured_text,
            }
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error("  Failed for '%s': %s", seed, exc)
