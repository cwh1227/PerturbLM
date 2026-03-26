"""CLI entrypoint for BMG subgraph extraction."""

import argparse

from .config import CACHE_DIR
from .generator import LLMConfig
from .logging_utils import configure_logging
from .pipeline import run_batch, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BMG-Conn Subgraph Extractor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=str, help="Single seed gene/drug name")
    group.add_argument("--batch", type=str, help="File with one seed per line")

    parser.add_argument("--hops", type=int, default=2, help="Max traversal hops (default: 2)")
    parser.add_argument("--max_nodes", type=int, default=60, help="Max nodes in subgraph (default: 60)")
    parser.add_argument(
        "--edge_types",
        type=str,
        default=None,
        help="Comma-separated edge types to include. Default: biological traversal path "
        "(Gene-Transcript,Transcript-Protein,...)",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "openai_compatible"],
        help="LLM API provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Model name for the selected LLM provider",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Optional API base URL, useful for OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key. If omitted, env var lookup is used.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default=None,
        help="Environment variable name to read the API key from",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens for LLM generation",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file for training data")
    parser.add_argument("--rebuild_cache", action="store_true", help="Force rebuild of cached indices")
    parser.add_argument("--dry_run", action="store_true", help="Skip LLM call; output raw graph structure only")
    parser.add_argument(
        "--no_bidirectional",
        action="store_true",
        help="Disable reverse-edge traversal (Drug-Protein etc. become one-directional again)",
    )
    return parser


def main() -> None:
    log = configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.rebuild_cache:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        log.info("Cache cleared")

    edge_types = args.edge_types.split(",") if args.edge_types else None
    llm_config = LLMConfig(
        provider=args.llm_provider,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.api_base,
        max_tokens=args.max_tokens,
    )

    bidirectional = not args.no_bidirectional

    if args.seed:
        run_pipeline(
            log=log,
            seed=args.seed,
            hops=args.hops,
            max_nodes=args.max_nodes,
            edge_types=edge_types,
            llm_config=llm_config,
            output_file=args.output,
            dry_run=args.dry_run,
            bidirectional=bidirectional,
        )
        return

    out = args.output or "training_data.jsonl"
    run_batch(
        log=log,
        input_file=args.batch,
        hops=args.hops,
        max_nodes=args.max_nodes,
        edge_types=edge_types,
        llm_config=llm_config,
        output_file=out,
        dry_run=args.dry_run,
        bidirectional=bidirectional,
    )
