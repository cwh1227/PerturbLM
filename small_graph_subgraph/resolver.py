"""Seed resolution helpers for the smaller_graph subgraph package."""

from .config import TYPE_PRIORITY


def _fuzzy_candidates(normalized: str, name_to_ids: dict[str, list[str]]) -> list[str]:
    """Return candidate IDs for substring matches, ranked by match quality.

    Ranking (lower is better):
      0 — exact match (already handled by caller, included for safety)
      1 — normalized appears as a whole word at a word boundary in the name
      2 — normalized matches at the start of the name
      3 — normalized appears anywhere in the name (shortest name wins ties)
    """
    scored: list[tuple[int, int, str]] = []  # (rank, name_len, name)
    for name, ids in name_to_ids.items():
        if normalized not in name:
            continue
        if name == normalized:
            rank = 0
        elif (f" {normalized} " in f" {name} "):
            # whole-word boundary match
            rank = 1
        elif name.startswith(normalized):
            rank = 2
        else:
            rank = 3
        scored.append((rank, len(name), name, ids))

    scored.sort(key=lambda item: (item[0], item[1]))
    seen: dict[str, None] = {}
    for _, _, _, ids in scored:
        for node_id in ids:
            seen[node_id] = None
    return list(seen)


def resolve_seed(
    seed: str,
    name_to_ids: dict[str, list[str]],
    id_to_name: dict[str, str],
    id_to_type: dict[str, str],
) -> tuple[list[str], str, str]:
    """Resolve a seed name or ID to one or more node IDs."""
    normalized = seed.strip().lower()
    if not normalized:
        raise ValueError("Seed must be a non-empty string")

    # 1. Exact lookup
    candidates = list(dict.fromkeys(name_to_ids.get(normalized, [])))

    # 2. Ranked fuzzy substring lookup
    if not candidates:
        candidates = _fuzzy_candidates(normalized, name_to_ids)

    # 3. Raw ID fallback
    if not candidates and seed in id_to_name:
        candidates = [seed]

    if not candidates:
        raise ValueError(f"Seed '{seed}' not found in smaller_graph entity index")

    for preferred_type in TYPE_PRIORITY:
        typed = [c for c in candidates if id_to_type.get(c) == preferred_type]
        if typed:
            best = typed[0]
            return [best], id_to_name.get(best, best), id_to_type.get(best, "Unknown")

    best = candidates[0]
    return [best], id_to_name.get(best, best), id_to_type.get(best, "Unknown")
