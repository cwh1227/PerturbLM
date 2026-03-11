"""
CLI entry point for the genomics data-processing agent.

Usage:
    # Anthropic Claude (default)
    python -m Agent.main --files cells.h5ad hgnc_complete_set.txt --out_dir out_SC

    # OpenAI GPT-4o
    python -m Agent.main --files ... --provider openai --model gpt-4o --out_dir out_SC

    # Qwen (Aliyun DashScope)
    python -m Agent.main --files ... --provider qwen --model qwen-max --out_dir out_SC

    # DeepSeek
    python -m Agent.main --files ... --provider deepseek --model deepseek-chat --out_dir out_SC

    # Run with YAML config
    python -m Agent.main --config config.yaml

    # Draw the graph topology
    python -m Agent.main --draw

Config YAML format:
    files:
      - /path/to/data.h5ad
      - /path/to/hgnc_complete_set.txt
    out_dir: out_SC
    provider: anthropic          # anthropic | openai | qwen | deepseek
    model: claude-sonnet-4-6
    instructions: "batch_key is 'Study', use n_top_genes=4000"

Required environment variables (per provider):
    anthropic  → ANTHROPIC_API_KEY
    openai     → OPENAI_API_KEY
    qwen       → DASHSCOPE_API_KEY
    deepseek   → DEEPSEEK_API_KEY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from Agent.graph import SYSTEM_PROMPT, build_graph
from Agent.state import AgentState


def _make_prompt(files: list, out_dir: str, instructions: str = "") -> str:
    lines = [
        f"Process the following genomics data files and write output to: {out_dir}",
        "",
        "Input files:",
        *[f"  - {f}" for f in files],
        "",
        "Start by inspecting the files to understand their format, then execute "
        "the appropriate pipeline steps.",
    ]
    if instructions:
        lines += ["", f"Additional instructions: {instructions}"]
    return "\n".join(lines)


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


def parser_default_provider() -> str:
    return "anthropic"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genomics data-processing agent  (LangGraph + Claude)"
    )
    parser.add_argument("--files", nargs="+", help="Input file path(s)")
    parser.add_argument("--out_dir", default="out_agent", help="Output directory")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--instructions", default="", help="Extra instructions for the agent")
    parser.add_argument("--provider", default=parser_default_provider(),
                        choices=["anthropic", "openai", "qwen", "deepseek"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default="",
                        help="Model ID (default per provider: claude-sonnet-4-6 / gpt-4o / qwen-max / deepseek-chat)")
    parser.add_argument("--draw", action="store_true", help="Print graph topology and exit")
    args = parser.parse_args()

    # Resolve config
    files, out_dir, instructions = args.files or [], args.out_dir, args.instructions
    cfg: dict[str, Any] | None = None
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        files        = cfg.get("files", files)
        out_dir      = cfg.get("out_dir", out_dir)
        instructions = cfg.get("instructions", instructions)

    provider, model = _resolve_provider_and_model(args, cfg)
    graph = build_graph(model=model, provider=provider)

    if args.draw:
        try:
            print(graph.get_graph().draw_mermaid())
        except Exception:
            print("Nodes:", list(graph.get_graph().nodes))
        return

    if not files:
        parser.error("Provide files via --files or a YAML config with 'files' key.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n[AGENT] Starting  |  provider={provider}  model={model}  out_dir={out_dir}")
    print(f"[AGENT] Files: {files}\n")

    initial: AgentState = {
        "messages": [HumanMessage(content=SYSTEM_PROMPT + "\n\n" + _make_prompt(files, out_dir, instructions))],
        "out_dir": out_dir,
        "intermediate_dir": "",
    }

    for chunk in graph.stream(initial, stream_mode="values"):
        last = chunk["messages"][-1]
        if isinstance(last, AIMessage):
            if last.tool_calls:
                for tc in last.tool_calls:
                    args_preview = json.dumps(tc["args"], ensure_ascii=False)
                    if len(args_preview) > 200:
                        args_preview = args_preview[:200] + "..."
                    print(f"\n[AGENT] → {tc['name']}")
                    print(f"         {args_preview}")
            else:
                print(f"\n[AGENT] {last.content}")
        elif isinstance(last, ToolMessage):
            try:
                parsed = json.loads(last.content)
                if "error" in parsed:
                    print(f"[TOOL ERROR] {last.name}: {str(parsed['error'])[:400]}")
                else:
                    print(f"[TOOL OK]    {last.name}: {json.dumps(parsed)[:300]}")
            except Exception:
                print(f"[TOOL]       {last.name}: {str(last.content)[:300]}")

    print("\n[AGENT] Done.\n")


if __name__ == "__main__":
    main()
