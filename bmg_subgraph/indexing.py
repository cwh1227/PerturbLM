"""Index builders for entities and relations."""

import pickle
import re
from collections import defaultdict
from typing import Optional

import pandas as pd

from .config import BIDIRECTIONAL_EDGE_TYPES, CACHE_DIR, DESCRIPTION_FILES, DISPLAY_NAME_FILES, NAME_COL, RELATION_CSV

# Bump this when the cached tuple structure changes
_ENTITY_INDEX_VERSION = 3

# Strip leading "Source: " style prefixes (e.g. "NCBI Gene: ", "ChEBI: ")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9 _]+:\s*")
# Strip trailing Ensembl source annotation e.g. " [Source:HGNC Symbol;Acc:HGNC:11998]"
_ENSEMBL_TAG_RE = re.compile(r"\s*\[Source:[^\]]+\]")
# Strip " | ..." secondary field separators added by the combined files
_PIPE_RE = re.compile(r"\s*\|.*$", re.DOTALL)


def _clean_desc(raw: str, max_chars: int = 220) -> str:
    """Strip source prefixes/tags and truncate to first sentence or max_chars."""
    text = raw.strip()
    # Remove "NCBI Gene: " / "Ensembl: " / "ChEBI: " prefix from combined files
    text = _PREFIX_RE.sub("", text)
    # Remove Ensembl "[Source:...]" annotation
    text = _ENSEMBL_TAG_RE.sub("", text)
    # Remove pipe-separated secondary fields (combined files have "desc1 | desc2")
    text = _PIPE_RE.sub("", text).strip()
    # Truncate at first sentence boundary if within range
    dot = text.find(". ")
    if 20 < dot < max_chars:
        return text[: dot + 1]
    return text[:max_chars]


def build_entity_index(log) -> tuple[dict, dict, dict, dict]:
    """
    Returns:
        id_to_name: BMGC_ID -> display name
        id_to_type: BMGC_ID -> entity type  (Gene | Drug | Protein | ...)
        name_to_ids: lowercase name -> list of BMGC_IDs
        id_to_desc: BMGC_ID -> short description (empty string if unavailable)
    """
    cache_file = CACHE_DIR / "entity_index.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as handle:
            cached = pickle.load(handle)
        # Invalidate old cache if structure changed
        if isinstance(cached, tuple) and len(cached) == 5 and cached[0] == _ENTITY_INDEX_VERSION:
            log.info("Loading entity index from cache...")
            return cached[1], cached[2], cached[3], cached[4]
        log.info("Entity index cache is stale, rebuilding...")

    log.info("Building entity index from CSVs...")
    id_to_name: dict[str, str] = {}
    id_to_type: dict[str, str] = {}
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    id_to_desc: dict[str, str] = {}

    # --- Display names & types ---
    for entity_type, file_path in DISPLAY_NAME_FILES.items():
        if not file_path.exists():
            log.warning("Missing: %s", file_path)
            continue
        name_col = NAME_COL[entity_type]
        df = pd.read_csv(file_path, usecols=["BioMedGraphica_Conn_ID", name_col], dtype=str)
        df[name_col] = df[name_col].fillna("").str.strip()
        id_to_name.update(df.set_index("BioMedGraphica_Conn_ID")[name_col].to_dict())
        for bmgc_id in df["BioMedGraphica_Conn_ID"]:
            id_to_type[bmgc_id] = entity_type
        for _, row in df[df[name_col] != ""].iterrows():
            name_to_ids[row[name_col].lower()].append(row["BioMedGraphica_Conn_ID"])

    # --- Descriptions ---
    for entity_type, file_path in DESCRIPTION_FILES.items():
        if not file_path.exists():
            log.warning("Missing description file: %s", file_path)
            continue
        df = pd.read_csv(file_path, usecols=["BioMedGraphica_Conn_ID", "Description"], dtype=str)
        df["Description"] = df["Description"].fillna("")
        for _, row in df[df["Description"] != ""].iterrows():
            id_to_desc[row["BioMedGraphica_Conn_ID"]] = _clean_desc(row["Description"])

    result = (_ENTITY_INDEX_VERSION, id_to_name, id_to_type, dict(name_to_ids), id_to_desc)
    with open(cache_file, "wb") as handle:
        pickle.dump(result, handle)
    log.info("Entity index: %s entities, %s with descriptions", f"{len(id_to_name):,}", f"{len(id_to_desc):,}")
    return id_to_name, id_to_type, dict(name_to_ids), id_to_desc


def build_graph_index(
    log,
    allowed_edge_types: Optional[set[str]] = None,
    bidirectional: bool = True,
) -> dict:
    """Builds adjacency list BMGC_ID -> list[(neighbor_BMGC_ID, edge_type)].

    When ``bidirectional=True`` (default), edges listed in
    BIDIRECTIONAL_EDGE_TYPES are also stored in reverse so that BFS can
    traverse them in both directions.  Reverse edge types are named by
    flipping the two parts, e.g. Drug-Protein → Protein-Drug.
    Protein-Protein is excluded from this set to avoid doubling 16 M edges.
    """
    bidir_active = BIDIRECTIONAL_EDGE_TYPES if bidirectional else set()

    # Derive the full set of edge types that will appear in the adjacency list
    # (originals + any synthesised reverses) so we can build a stable cache key.
    effective_types = set(allowed_edge_types) if allowed_edge_types else set()
    reverse_map: dict[str, str] = {}  # original_type → reversed_type
    for etype in list(effective_types or bidir_active):
        if etype in bidir_active:
            parts = etype.split("-", 1)
            rev = f"{parts[1]}-{parts[0]}" if len(parts) == 2 else f"REV-{etype}"
            reverse_map[etype] = rev

    suffix = "bidir" if (bidirectional and reverse_map) else "norev"
    if allowed_edge_types:
        types_key = "_".join(sorted(allowed_edge_types))[:55]
        cache_file = CACHE_DIR / f"graph_{types_key}_{suffix}.pkl"
    else:
        cache_file = CACHE_DIR / f"graph_full_{suffix}.pkl"

    if cache_file.exists():
        log.info("Loading graph index from cache...")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    log.info("Building graph index from relation CSV (this takes a few minutes)...")
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    chunk_size = 1_000_000
    total_rows = 0

    for chunk in pd.read_csv(
        RELATION_CSV,
        usecols=["BMGC_From_ID", "BMGC_To_ID", "Type"],
        chunksize=chunk_size,
        dtype=str,
    ):
        chunk = chunk.dropna(subset=["BMGC_From_ID", "BMGC_To_ID", "Type"])
        if allowed_edge_types:
            chunk = chunk[chunk["Type"].isin(allowed_edge_types)]

        chunk["pair"] = list(zip(chunk["BMGC_To_ID"], chunk["Type"]))
        for src, group in chunk.groupby("BMGC_From_ID"):
            adjacency[src].extend(group["pair"].tolist())

        # Synthesise reverse edges for bidirectional types
        if reverse_map:
            bidir_chunk = chunk[chunk["Type"].isin(reverse_map)]
            if not bidir_chunk.empty:
                for _, row in bidir_chunk.iterrows():
                    rev_type = reverse_map[row["Type"]]
                    adjacency[row["BMGC_To_ID"]].append((row["BMGC_From_ID"], rev_type))

        total_rows += len(chunk)
        log.info("  Processed %s filtered edges...", f"{total_rows:,}")

    graph = dict(adjacency)
    rev_count = sum(
        1 for neighbors in graph.values()
        for _, et in neighbors if et in reverse_map.values()
    )
    log.info(
        "Graph index built: %s source nodes (%s reverse edges added), saving to cache...",
        f"{len(graph):,}", f"{rev_count:,}",
    )
    with open(cache_file, "wb") as handle:
        pickle.dump(graph, handle)
    return graph
