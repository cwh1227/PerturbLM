"""
CLI entry point for ATAC key-feature extraction agent.

Usage:
    python -m Agent.main_atac --files peak_to_gene.tsv --out_dir out_atac
    python -m Agent.main_atac --files /path/to/Liscovitch-BrauerSanjana2021_K562_1 --out_dir out_atac
    python -m Agent.main_atac --config atac_config.yaml

Config YAML format:
    files:
      - /path/to/peak_to_gene.tsv
    out_dir: out_atac
    provider: anthropic
    model: claude-sonnet-4-6
    instructions: "gene_col is nearest_gene, distance_col is dist_to_tss"
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


try:
    _graph_mod = importlib.import_module("Agent.graph_atac")
    _state_mod = importlib.import_module("Agent.state")
except ModuleNotFoundError:
    _graph_mod = importlib.import_module("graph_atac")
    _state_mod = importlib.import_module("state")

SYSTEM_PROMPT_ATAC = _graph_mod.SYSTEM_PROMPT_ATAC
build_graph_atac = _graph_mod.build_graph_atac
AgentState = _state_mod.AgentState


def _make_prompt(files: list[str], out_dir: str, instructions: str = "") -> str:
    resolved_paths = [Path(f) for f in files]
    dataset_dirs = [str(p) for p in resolved_paths if p.exists() and p.is_dir()]
    data_files = [str(p) for p in resolved_paths if not (p.exists() and p.is_dir())]

    lines = [
        f"Process the following ATAC inputs and write gene-level output to: {out_dir}",
        "",
        "Input paths:",
        *[f"  - {f}" for f in files],
        "",
    ]

    if dataset_dirs:
        lines += [
            "Detected directory-style ATAC bundle input(s).",
            "Use bundle workflow with QC by default:",
            "1) inspect_atac_bundle(dataset_dir)",
            "2) compute_atac_gene_scores_from_bundle(dataset_dir, out_dir,",
            "   gene_col='nearestGene', distance_col='distToTSS',",
            "   min_tss_enrichment=4.0, min_frip=0.15,",
            "   min_fragments=3000, max_blacklist_ratio=0.05,",
            "   min_peak_cells=5, min_peaks_per_gene=3)",
        ]

    if data_files:
        lines += [
            "For flat table input(s), use:",
            "1) inspect_atac_table(path)",
            "2) compute_atac_gene_scores(path, out_dir, gene_col, distance_col, openness_col)",
        ]

    lines += [
        "",
        "Always inspect first, then compute per-gene composite_score.",
    ]

    if instructions:
        lines += ["", f"Additional instructions: {instructions}"]
    return "\n".join(lines)


def parser_default_provider() -> str:
    return "anthropic"


def _resolve_provider_and_model(args: Any, cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    default_models = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "qwen": "qwen-max",
        "deepseek": "deepseek-chat",
    }

    provider = args.provider
    if cfg and args.provider == parser_default_provider():
        provider = cfg.get("provider", provider)

    if args.model:
        model = args.model
    elif cfg and cfg.get("model"):
        model = cfg["model"]
    else:
        model = default_models[provider]

    return provider, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATAC key-feature extraction agent (LangGraph + ReAct)"
    )
    parser.add_argument("--files", nargs="+", help="Input file path(s)")
    parser.add_argument("--out_dir", default="out_atac", help="Output directory")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--instructions", default="", help="Extra instructions for the agent")
    parser.add_argument(
        "--provider",
        default=parser_default_provider(),
        choices=["anthropic", "openai", "qwen", "deepseek"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model ID (default per provider)",
    )
    parser.add_argument("--draw", action="store_true", help="Print graph topology and exit")
    args = parser.parse_args()

    files, out_dir, instructions = args.files or [], args.out_dir, args.instructions
    cfg: dict[str, Any] | None = None

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        files = cfg.get("files", files)
        out_dir = cfg.get("out_dir", out_dir)
        instructions = cfg.get("instructions", instructions)

    provider, model = _resolve_provider_and_model(args, cfg)
    graph = build_graph_atac(model=model, provider=provider)

    if args.draw:
        try:
            print(graph.get_graph().draw_mermaid())
        except Exception:
            print("Nodes:", list(graph.get_graph().nodes))
        return

    if not files:
        parser.error("Provide files via --files or a YAML config with 'files' key.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n[ATAC AGENT] Starting  |  provider={provider}  model={model}  out_dir={out_dir}")
    print(f"[ATAC AGENT] Files: {files}\n")

    initial: AgentState = {
        "messages": [
            HumanMessage(content=SYSTEM_PROMPT_ATAC + "\n\n" + _make_prompt(files, out_dir, instructions))
        ],
        "out_dir": out_dir,
        "intermediate_dir": "",
    }

    for chunk in graph.stream(initial, stream_mode="values"):
        last = chunk["messages"][-1]
        if isinstance(last, AIMessage):
            if last.tool_calls:
                for tc in last.tool_calls:
                    args_preview = json.dumps(tc["args"], ensure_ascii=False)
                    if len(args_preview) > 240:
                        args_preview = args_preview[:240] + "..."
                    print(f"\n[ATAC AGENT] → {tc['name']}")
                    print(f"               {args_preview}")
            else:
                print(f"\n[ATAC AGENT] {last.content}")
        elif isinstance(last, ToolMessage):
            try:
                parsed = json.loads(last.content)
                if "error" in parsed:
                    print(f"[TOOL ERROR] {last.name}: {str(parsed['error'])[:400]}")
                else:
                    print(f"[TOOL OK]    {last.name}: {json.dumps(parsed, ensure_ascii=False)[:320]}")
            except Exception:
                print(f"[TOOL]       {last.name}: {str(last.content)[:320]}")

    print("\n[ATAC AGENT] Done.\n")


if __name__ == "__main__":
    main()
