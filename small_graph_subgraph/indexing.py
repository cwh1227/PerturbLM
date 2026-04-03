"""Index builders for the smaller_graph knowledge graph."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    CACHE_DIR,
    DEFAULT_EDGE_TYPES,
    ENTITY_INDEX_CANDIDATES,
    HGNC_FILE,
    KG_EDGE_PARQUET_CANDIDATES,
    KG_TRIPLES_CANDIDATES,
    RELATION_INDEX_CANDIDATES,
    TYPE_PREFIX_TO_NAME,
)

_ENTITY_INDEX_VERSION = 2
_GRAPH_INDEX_VERSION = 2


def _shorten(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    dot = cleaned.find(". ")
    if 20 < dot < max_chars:
        return cleaned[: dot + 1]
    return cleaned[:max_chars]


def _find_existing(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_entity_id(entity_id: str) -> tuple[str, str]:
    prefix, raw_id = entity_id.split(":", 1)
    return prefix, raw_id


def _display_name(entity_id: str) -> tuple[str, str]:
    if ":" not in entity_id:
        return "Unknown", entity_id
    prefix, raw_id = _split_entity_id(entity_id)
    return TYPE_PREFIX_TO_NAME.get(prefix, prefix.title()), raw_id


def _load_gene_descriptions() -> dict[str, str]:
    if not HGNC_FILE.exists():
        return {}

    hgnc = pd.read_csv(HGNC_FILE, sep="\t", dtype=str, low_memory=False).fillna("")
    gene_desc: dict[str, str] = {}
    for _, row in hgnc.iterrows():
        symbol = row.get("symbol", "").strip()
        if not symbol:
            continue
        desc = row.get("name", "").strip()
        if desc:
            gene_desc[symbol] = _shorten(desc)
    return gene_desc


def _load_known_relations(log) -> list[str]:
    relation_file = _find_existing(RELATION_INDEX_CANDIDATES)
    if not relation_file:
        return list(DEFAULT_EDGE_TYPES)

    payload = _load_json(relation_file)
    if isinstance(payload, dict):
        relations = list(payload.keys())
        log.info("Loaded %s relation types from %s", len(relations), relation_file)
        return relations
    return list(DEFAULT_EDGE_TYPES)


def build_entity_index(log) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Build generic entity maps for the new gene-centric small graph."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "entity_index.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as handle:
            cached = pickle.load(handle)
        if isinstance(cached, tuple) and len(cached) == 5 and cached[0] == _ENTITY_INDEX_VERSION:
            log.info("Loading smaller_graph entity index from cache...")
            return cached[1], cached[2], cached[3], cached[4]
        log.info("smaller_graph entity index cache is stale, rebuilding...")

    entity_file = _find_existing(ENTITY_INDEX_CANDIDATES)
    if not entity_file:
        checked = ", ".join(str(path) for path in ENTITY_INDEX_CANDIDATES)
        raise FileNotFoundError(f"No entity index found. Checked: {checked}")

    log.info("Building smaller_graph entity index from %s...", entity_file)
    entity_payload = _load_json(entity_file)
    entity_ids = list(entity_payload.keys()) if isinstance(entity_payload, dict) else list(entity_payload)

    gene_desc = _load_gene_descriptions()
    id_to_name: dict[str, str] = {}
    id_to_type: dict[str, str] = {}
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    id_to_desc: dict[str, str] = {}

    for entity_id in entity_ids:
        entity_type, display_name = _display_name(entity_id)
        id_to_name[entity_id] = display_name
        id_to_type[entity_id] = entity_type

        keys = {entity_id.lower(), display_name.lower()}
        if ":" in entity_id:
            _, raw_id = _split_entity_id(entity_id)
            keys.add(raw_id.lower())
            if entity_type == "Gene":
                desc = gene_desc.get(raw_id)
                if desc:
                    id_to_desc[entity_id] = desc

        for key in keys:
            name_to_ids[key].append(entity_id)

    result = (_ENTITY_INDEX_VERSION, id_to_name, id_to_type, dict(name_to_ids), id_to_desc)
    with cache_file.open("wb") as handle:
        pickle.dump(result, handle)

    log.info(
        "smaller_graph entity index: %s entities, %s with descriptions",
        f"{len(id_to_name):,}",
        f"{len(id_to_desc):,}",
    )
    return id_to_name, id_to_type, dict(name_to_ids), id_to_desc


def _load_edge_frame(log) -> pd.DataFrame:
    parquet_file = _find_existing(KG_EDGE_PARQUET_CANDIDATES)
    if parquet_file:
        log.info("Loading KG edges from %s", parquet_file)
        edge_df = pd.read_parquet(parquet_file)
        expected = {"source", "target", "edge_type"}
        missing = expected.difference(edge_df.columns)
        if missing:
            raise ValueError(f"Missing columns in {parquet_file}: {sorted(missing)}")
        return edge_df[["source", "target", "edge_type"]].astype(str)

    triple_file = _find_existing(KG_TRIPLES_CANDIDATES)
    if triple_file:
        log.info("Loading KG triples from %s", triple_file)
        return pd.read_csv(
            triple_file,
            sep="\t",
            header=None,
            names=["source", "edge_type", "target"],
            dtype=str,
        ).fillna("")

    checked = [*KG_EDGE_PARQUET_CANDIDATES, *KG_TRIPLES_CANDIDATES]
    checked_str = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "No raw small_graph edge file found. Expected kg_edges.parquet or kg_triples.tsv. "
        f"Checked: {checked_str}"
    )


def build_graph_index(
    log,
    allowed_edge_types: Optional[set[str]] = None,
    bidirectional: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    """Build adjacency list for the new gene-centric small graph."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    known_relations = _load_known_relations(log)
    allowed = set(allowed_edge_types) if allowed_edge_types else set(known_relations)
    allowed_key = "all" if not allowed else "_".join(sorted(allowed))
    suffix = "bidir" if bidirectional else "norev"
    cache_file = CACHE_DIR / f"graph_v{_GRAPH_INDEX_VERSION}_{allowed_key}_{suffix}.pkl"

    if cache_file.exists():
        log.info("Loading smaller_graph adjacency from cache...")
        with cache_file.open("rb") as handle:
            return pickle.load(handle)

    log.info("Building smaller_graph adjacency index...")
    edge_df = _load_edge_frame(log)
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def wants(edge_type: str) -> bool:
        return edge_type in allowed

    def add_edge(src: str, dst: str, edge_type: str) -> None:
        adjacency[src].append((dst, edge_type))

    edge_count = 0
    for row in edge_df.itertuples(index=False):
        src = str(row.source).strip()
        dst = str(row.target).strip()
        edge_type = str(row.edge_type).strip()
        if not src or not dst or not edge_type:
            continue
        if not wants(edge_type):
            continue

        add_edge(src, dst, edge_type)
        edge_count += 1
        if bidirectional:
            add_edge(dst, src, f"rev_{edge_type}")

    graph = dict(adjacency)
    with cache_file.open("wb") as handle:
        pickle.dump(graph, handle)

    log.info(
        "smaller_graph adjacency built for %s source nodes across %s edges",
        f"{len(graph):,}",
        f"{edge_count:,}",
    )
    return graph
