"""Structured text generation utilities."""

from dataclasses import dataclass
import os
from typing import Any, Optional

SYSTEM_PROMPT = """You are a biomedical knowledge graph expert.
Given a subgraph extracted from the BioMedGraphica-Conn knowledge graph,
generate a structured text representation suitable for LLM training.

Rules:
1. Output the <|HVG_subgraph|> block exactly as specified
2. For NODES: each node must have TWO labels separated by |:
   - First label: entity category  (Gene | Drug | Protein | Transcript | Disease | Pathway | Phenotype | Metabolite | Exposure | Microbiota | Promoter)
   - Second label: concise biological role derived from the description
     (e.g., tumor suppressor, kinase inhibitor, cell cycle regulator, TF, oncogene,
      receptor tyrosine kinase, apoptosis regulator, metabolic enzyme, etc.)
   Format: NodeName [Category | biological role]
   Example: TP53 [Gene | tumor suppressor]
            Imatinib [Drug | BCR-ABL inhibitor]
            MDM2 [Protein | p53 negative regulator]
   If no meaningful role can be determined, use the category alone: NodeName [Gene]
3. For EDGES: use biological relationship verbs
   (activates, inhibits, encodes, transcribes, phosphorylates, binds, regulates,
   interacts with, associated with, metabolized by, targets, etc.)
4. Only include nodes and edges that have clear biological meaning
5. Keep node names concise (use gene symbols, not full names when possible)
6. The output must be valid structured text — no extra commentary"""

USER_TEMPLATE = """Seed entity: {seed_name} ({seed_type})

Raw subgraph nodes (Name [Type] | description):
{node_list}

Raw subgraph edges:
{edge_list}

Generate the structured subgraph text in this exact format:

<|HVG_subgraph|>
NODES:
<node_name> [Category | biological role]
...

EDGES:
<node_a> -> <node_b> (relationship)
...
"""


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-4-6"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048


def format_raw_subgraph(nodes: dict, edges: list) -> str:
    """Produce a minimal structured text without LLM.

    Nodes are labelled [Category | description] when a description is
    available, otherwise just [Category].
    """
    lines = ["<|HVG_subgraph|>", "NODES:"]
    for node in nodes.values():
        desc = node.get("desc", "").strip()
        label = f"{node['type']} | {desc}" if desc else node["type"]
        lines.append(f"{node['name']} [{label}]")
    lines.append("\nEDGES:")
    for edge in edges:
        src_name = nodes.get(edge["src"], {}).get("name", edge["src"])
        dst_name = nodes.get(edge["dst"], {}).get("name", edge["dst"])
        lines.append(f"{src_name} -> {dst_name} ({edge['type']})")
    return "\n".join(lines)


def _resolve_api_key(config: LLMConfig, default_env: str) -> Optional[str]:
    if config.api_key:
        return config.api_key
    if config.api_key_env:
        return os.getenv(config.api_key_env)
    return os.getenv(default_env)


def create_llm_client(config: LLMConfig) -> Any:
    """Create an LLM client for the selected provider."""
    if config.provider == "anthropic":
        import anthropic

        api_key = _resolve_api_key(config, "ANTHROPIC_API_KEY")
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        return anthropic.Anthropic(**client_kwargs)

    if config.provider in {"openai", "openai_compatible"}:
        from openai import OpenAI

        api_key = _resolve_api_key(config, "OPENAI_API_KEY")
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        return OpenAI(**client_kwargs)

    raise ValueError(
        "Unsupported LLM provider '%s'. Use one of: anthropic, openai, openai_compatible"
        % config.provider
    )


def generate_subgraph_text(
    seed_name: str,
    seed_type: str,
    nodes: dict,
    edges: list,
    client: Any,
    config: LLMConfig,
) -> str:
    """Call the selected LLM provider to format the subgraph as structured training text."""
    def _node_line(node: dict) -> str:
        base = f"  {node['name']} [{node['type']}]"
        desc = node.get("desc", "").strip()
        return f"{base} | {desc}" if desc else base

    node_list = "\n".join(_node_line(node) for node in nodes.values())
    edge_list = "\n".join(
        f"  {nodes.get(edge['src'], {}).get('name', edge['src'])} --[{edge['type']}]--> "
        f"{nodes.get(edge['dst'], {}).get('name', edge['dst'])}"
        for edge in edges
    )

    user_msg = USER_TEMPLATE.format(
        seed_name=seed_name,
        seed_type=seed_type,
        node_list=node_list,
        edge_list=edge_list,
    )

    if config.provider == "anthropic":
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()

    if config.provider in {"openai", "openai_compatible"}:
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content.strip()

    raise ValueError("Unsupported LLM provider '%s'" % config.provider)
