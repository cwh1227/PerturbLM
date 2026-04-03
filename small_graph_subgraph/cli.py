"""CLI entrypoint for smaller_graph subgraph extraction."""

import argparse

from .config import CACHE_DIR, DEFAULT_EDGE_TYPES
from .pipeline import run_batch, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="smaller_graph subgraph extractor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=str, help="Single seed name or node ID")
    group.add_argument("--batch", type=str, help="Text file with one seed per line")

    parser.add_argument("--hops", type=int, default=2, help="Max traversal hops (default: 2)")
    parser.add_argument("--max_nodes", type=int, default=60, help="Max nodes in subgraph (default: 60)")
    parser.add_argument(
        "--edge_types",
        type=str,
        default=None,
        help=(
            "Comma-separated edge types to include. Default: "
            + ",".join(DEFAULT_EDGE_TYPES)
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSONL output path")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild cached indices")
    parser.add_argument(
        "--no_bidirectional",
        action="store_true",
        help="Disable automatic reverse traversal via rev_<relation> edges",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Attach pre-trained KG node embeddings to each result "
            "(HeteroGraphSAGE 4-layer preferred, RotatE fallback). "
            "Stored under 'node_embeddings' in the JSONL output."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.rebuild_cache:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()

    edge_types = [item.strip() for item in args.edge_types.split(",")] if args.edge_types else None
    bidirectional = not args.no_bidirectional

    if args.seed:
        run_pipeline(
            seed=args.seed,
            hops=args.hops,
            max_nodes=args.max_nodes,
            edge_types=edge_types,
            output_file=args.output,
            bidirectional=bidirectional,
            embed=args.embed,
        )
        return

    run_batch(
        input_file=args.batch,
        hops=args.hops,
        max_nodes=args.max_nodes,
        edge_types=edge_types,
        output_file=args.output or "small_graph_subgraphs.jsonl",
        bidirectional=bidirectional,
        embed=args.embed,
    )


if __name__ == "__main__":
    main()
