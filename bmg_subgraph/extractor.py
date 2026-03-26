"""Subgraph extraction logic."""

import math
from collections import defaultdict
from typing import Optional


# Ratios are relative to max_nodes. min/max keep the budgets sane when users
# run very small or very large extractions.
DEFAULT_TYPE_BUDGETS: dict[str, dict[str, float | int]] = {
    "Transcript": {"ratio": 0.10, "min": 3, "max": 12},
    "Protein": {"ratio": 0.30, "min": 10, "max": 40},
    "Drug": {"ratio": 0.16, "min": 4, "max": 24},
    "Disease": {"ratio": 0.14, "min": 4, "max": 20},
    "Pathway": {"ratio": 0.10, "min": 3, "max": 15},
    "Phenotype": {"ratio": 0.10, "min": 3, "max": 15},
    "Metabolite": {"ratio": 0.10, "min": 3, "max": 15},
    "Exposure": {"ratio": 0.06, "min": 2, "max": 10},
    "Gene": {"ratio": 0.12, "min": 3, "max": 20},
    "Microbiota": {"ratio": 0.06, "min": 2, "max": 10},
    "Promoter": {"ratio": 0.05, "min": 2, "max": 8},
    "*": {"ratio": 0.08, "min": 2, "max": 12},
}

TYPE_PRIORITY_BONUS: dict[str, float] = {
    "Drug": 2.5,
    "Disease": 2.2,
    "Pathway": 1.8,
    "Phenotype": 1.6,
    "Metabolite": 1.4,
    "Gene": 1.2,
    "Protein": 0.6,
    "Transcript": 0.0,
}

EDGE_PRIORITY_BONUS: dict[str, float] = {
    "Drug-Protein": 2.6,
    "Protein-Drug": 2.6,
    "Drug-Disease": 2.4,
    "Disease-Drug": 2.4,
    "Protein-Disease": 2.0,
    "Disease-Protein": 2.0,
    "Protein-Pathway": 1.7,
    "Pathway-Protein": 1.7,
    "Protein-Phenotype": 1.5,
    "Phenotype-Protein": 1.5,
    "Metabolite-Protein": 1.4,
    "Protein-Metabolite": 1.4,
    "Gene-Transcript": 0.6,
    "Transcript-Protein": 0.8,
    "Protein-Protein": -0.2,
}


def _compute_type_caps(max_nodes: int, type_budgets: Optional[dict[str, dict[str, float | int]]] = None) -> dict[str, int]:
    budgets = type_budgets or DEFAULT_TYPE_BUDGETS
    caps: dict[str, int] = {}
    for entity_type, rule in budgets.items():
        ratio = float(rule.get("ratio", 0.0))
        min_cap = int(rule.get("min", 0))
        max_cap = int(rule.get("max", max_nodes))
        computed = max(min_cap, int(round(max_nodes * ratio)))
        caps[entity_type] = min(computed, max_cap, max_nodes)
    caps.setdefault("*", max(2, min(max_nodes, int(round(max_nodes * 0.08)))))
    return caps


def _candidate_score(node: dict, edge_type: str, type_counts: dict[str, int]) -> float:
    entity_type = node["type"]
    score = TYPE_PRIORITY_BONUS.get(entity_type, 0.4)
    score += EDGE_PRIORITY_BONUS.get(edge_type, 0.0)
    score += 0.8 if node.get("desc", "").strip() else 0.0
    score += max(0.0, 1.8 - 0.35 * type_counts.get(entity_type, 0))
    return score


def _select_candidates(
    candidates: dict[str, dict],
    type_counts: dict[str, int],
    type_caps: dict[str, int],
    remaining_global_budget: int,
    remaining_hops: int,
) -> list[str]:
    if remaining_global_budget <= 0 or not candidates:
        return []

    candidates_by_type: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates.values():
        candidates_by_type[candidate["node"]["type"]].append(candidate)

    for bucket in candidates_by_type.values():
        bucket.sort(key=lambda item: item["score"], reverse=True)

    selected: list[str] = []
    selected_ids: set[str] = set()
    per_hop_counts: dict[str, int] = defaultdict(int)

    # First pass: reserve at least one slot for underrepresented types when possible.
    underfilled_types = sorted(
        candidates_by_type,
        key=lambda entity_type: (
            type_counts.get(entity_type, 0),
            -TYPE_PRIORITY_BONUS.get(entity_type, 0.0),
        ),
    )
    for entity_type in underfilled_types:
        cap = type_caps.get(entity_type, type_caps.get("*", remaining_global_budget))
        if type_counts.get(entity_type, 0) >= cap:
            continue
        if len(selected) >= remaining_global_budget:
            break
        chosen = candidates_by_type[entity_type][0]
        selected.append(chosen["id"])
        selected_ids.add(chosen["id"])
        per_hop_counts[entity_type] += 1

    # Second pass: fill the rest by score while respecting per-type and per-hop budgets.
    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    for candidate in ranked:
        if len(selected) >= remaining_global_budget:
            break
        candidate_id = candidate["id"]
        if candidate_id in selected_ids:
            continue

        entity_type = candidate["node"]["type"]
        cap = type_caps.get(entity_type, type_caps.get("*", remaining_global_budget))
        remaining_type_budget = cap - type_counts.get(entity_type, 0) - per_hop_counts.get(entity_type, 0)
        if remaining_type_budget <= 0:
            continue

        per_hop_cap = max(1, math.ceil((cap - type_counts.get(entity_type, 0)) / max(1, remaining_hops)))
        if per_hop_counts.get(entity_type, 0) >= per_hop_cap:
            continue

        selected.append(candidate_id)
        selected_ids.add(candidate_id)
        per_hop_counts[entity_type] += 1

    return selected


def extract_subgraph(
    seed_ids: list[str],
    adj: dict,
    id_to_name: dict,
    id_to_type: dict,
    id_to_desc: dict,
    max_hops: int = 2,
    max_nodes: int = 60,
    allowed_edge_types: Optional[set[str]] = None,
    type_caps: Optional[dict[str, int]] = None,
) -> tuple[dict, list]:
    """Run layered BFS and balance expansion across entity types.

    The extractor expands hop-by-hop. Each hop first gathers candidate nodes,
    then selects a balanced subset using dynamic per-type budgets and a simple
    biological relevance score. This reduces domination by high-degree types
    such as Transcript and Protein while preserving a global max_nodes limit.
    """
    caps = type_caps if type_caps is not None else _compute_type_caps(max_nodes)
    type_counts: dict[str, int] = defaultdict(int)

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    visited: set[str] = set()

    def _make_node(bmgc_id: str) -> dict:
        return {
            "name": id_to_name.get(bmgc_id, bmgc_id),
            "type": id_to_type.get(bmgc_id, "Unknown"),
            "desc": id_to_desc.get(bmgc_id, ""),
        }

    def _record_edge(src: str, dst: str, edge_type: str) -> None:
        edge_key = (src, dst, edge_type)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edges.append({"src": src, "dst": dst, "type": edge_type})

    current_frontier: list[str] = []
    for seed_id in seed_ids:
        node = _make_node(seed_id)
        nodes[seed_id] = node
        type_counts[node["type"]] += 1
        visited.add(seed_id)
        current_frontier.append(seed_id)

    for hop in range(max_hops):
        if not current_frontier or len(nodes) >= max_nodes:
            break

        candidates: dict[str, dict] = {}
        pending_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for current in current_frontier:
            for neighbor, edge_type in adj.get(current, []):
                if allowed_edge_types and edge_type not in allowed_edge_types:
                    continue

                if neighbor in nodes:
                    _record_edge(current, neighbor, edge_type)
                    continue

                node = _make_node(neighbor)
                score = _candidate_score(node, edge_type, type_counts)
                best = candidates.get(neighbor)
                if best is None or score > best["score"]:
                    candidates[neighbor] = {
                        "id": neighbor,
                        "node": node,
                        "score": score,
                    }
                pending_edges[neighbor].append((current, edge_type))

        remaining_global_budget = max_nodes - len(nodes)
        selected_ids = _select_candidates(
            candidates=candidates,
            type_counts=type_counts,
            type_caps=caps,
            remaining_global_budget=remaining_global_budget,
            remaining_hops=max_hops - hop,
        )

        next_frontier: list[str] = []
        for node_id in selected_ids:
            candidate = candidates[node_id]
            node = candidate["node"]
            entity_type = node["type"]
            cap = caps.get(entity_type, caps.get("*", max_nodes))
            if type_counts.get(entity_type, 0) >= cap:
                continue

            nodes[node_id] = node
            type_counts[entity_type] += 1
            visited.add(node_id)
            next_frontier.append(node_id)

            for src, edge_type in pending_edges.get(node_id, []):
                if src in nodes:
                    _record_edge(src, node_id, edge_type)

        current_frontier = next_frontier

    node_ids = set(nodes)
    filtered_edges = [edge for edge in edges if edge["src"] in node_ids and edge["dst"] in node_ids]
    return nodes, filtered_edges

