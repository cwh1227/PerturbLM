"""Seed resolution helpers."""

from .config import TYPE_PRIORITY


def resolve_seed(
    seed: str,
    name_to_ids: dict,
    id_to_name: dict,
    id_to_type: dict,
) -> tuple[list[str], str, str]:
    """Resolve a seed name to BMGC IDs."""
    candidates = name_to_ids.get(seed.lower(), [])
    if not candidates:
        for name, ids in name_to_ids.items():
            if seed.lower() in name:
                candidates = ids
                break

    if not candidates:
        raise ValueError(f"Seed '{seed}' not found in entity index")

    for preferred_type in TYPE_PRIORITY:
        typed = [candidate for candidate in candidates if id_to_type.get(candidate) == preferred_type]
        if typed:
            best = typed[0]
            return [best], id_to_name[best], id_to_type[best]

    best = candidates[0]
    return [best], id_to_name.get(best, best), id_to_type.get(best, "Unknown")
