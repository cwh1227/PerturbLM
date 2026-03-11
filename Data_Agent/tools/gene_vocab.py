"""
Gene HGNC mapping tool.

Reads  intermediate/gene_meta.parquet
Writes intermediate/gene_meta.parquet  (adds hgnc_num, hgnc_token)
Saves  out_dir/gene_vocab.json, conn_id_to_hgnc.json, gene_mapping_table.tsv

Completely format-agnostic: works on the standardized gene_meta columns
(orig_id, gene_symbol, ensembl_id) written by any read_* tool.

Mapping priority:
  1. ensembl_id → HGNC ensembl_gene_id
  2. gene_symbol → HGNC symbol / alias_symbol / prev_symbol
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from Agent.state import gene_meta_path


def _extract_hgnc_num(s) -> str:
    s = str(s).strip()
    return s.split(":", 1)[-1] if pd.notna(s) and ":" in s else ""


def _norm_ensembl(x) -> str:
    s = str(x).strip()
    if s.lower() in ("", "nan", "na", "none"):
        return ""
    if s.startswith("ENS") and "." in s:
        s = s.split(".", 1)[0]
    return s


def _build_symbol_lookup(hgnc_df: pd.DataFrame) -> dict:
    valid  = hgnc_df[hgnc_df["hgnc_num"] != ""].copy()
    lookup: dict = {}
    sym = valid[["symbol", "hgnc_num"]].dropna(subset=["symbol"]).copy()
    sym["symbol"] = sym["symbol"].str.strip()
    sym = sym[sym["symbol"] != ""]
    lookup.update(zip(sym["symbol"], sym["hgnc_num"]))
    lookup.update(zip(sym["symbol"].str.upper(), sym["hgnc_num"]))
    for col in ("alias_symbol", "prev_symbol"):
        if col not in valid.columns:
            continue
        a = valid[["hgnc_num", col]].dropna(subset=[col]).copy()
        a[col] = a[col].str.strip().str.strip('"')
        a = a[a[col] != ""]
        a = a.assign(**{col: a[col].str.split("|")}).explode(col)
        a[col] = a[col].str.strip()
        a = a[a[col] != ""]
        lookup.update(zip(a[col], a["hgnc_num"]))
        lookup.update(zip(a[col].str.upper(), a["hgnc_num"]))
    return lookup


@tool
def map_to_hgnc(hgnc_path: str, out_dir: str, intermediate_dir: str = "") -> str:
    """
    Map all genes in gene_meta.parquet to HGNC numeric IDs and build the vocabulary.

    Reads gene_meta.parquet (produced by read_* tools), maps each gene via:
      1. ensembl_id → HGNC ensembl_gene_id  (primary)
      2. gene_symbol → HGNC symbol / alias / prev_symbol  (fallback)

    Only genes with is_hvg=True AND a valid HGNC ID are included in the vocabulary.
    All other genes get hgnc_token=0 and are silently dropped at tokenization time.

    Writes updated gene_meta.parquet with hgnc_num and hgnc_token columns.
    Saves out_dir/gene_vocab.json, conn_id_to_hgnc.json, gene_mapping_table.tsv.

    Returns JSON with vocab size and mapping statistics.
    """
    try:
        gene_meta = pd.read_parquet(gene_meta_path(intermediate_dir))
        out       = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Load HGNC reference ───────────────────────────────────────────
        hgnc_df = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
        hgnc_df["hgnc_num"]     = hgnc_df["hgnc_id"].apply(_extract_hgnc_num)
        hgnc_df["ensembl_norm"] = hgnc_df["ensembl_gene_id"].apply(_norm_ensembl)

        hgnc_ens = (
            hgnc_df
            .loc[(hgnc_df["ensembl_norm"] != "") & (hgnc_df["hgnc_num"] != ""),
                 ["ensembl_norm", "hgnc_num"]]
            .drop_duplicates("ensembl_norm", keep="first")
        )
        ens_to_hgnc    = dict(zip(hgnc_ens["ensembl_norm"], hgnc_ens["hgnc_num"]))
        symbol_lookup  = _build_symbol_lookup(hgnc_df)

        # ── Map each gene ─────────────────────────────────────────────────
        gm = gene_meta.copy()
        gm["_ens"] = gm["ensembl_id"].astype(str).apply(_norm_ensembl)

        gm["hgnc_num"] = gm["_ens"].map(ens_to_hgnc).fillna("")

        needs_fb = gm["hgnc_num"].str.lower().isin(["", "nan", "none"])
        if needs_fb.any():
            sym = gm.loc[needs_fb, "gene_symbol"].astype(str).str.strip()
            gm.loc[needs_fb, "hgnc_num"] = (
                sym.map(symbol_lookup)
                   .fillna(sym.str.upper().map(symbol_lookup))
                   .fillna("")
            )

        gm.drop(columns=["_ens"], inplace=True)

        # ── Build vocabulary (HVG + mapped genes only) ────────────────────
        valid_mask = (
            gm["is_hvg"]
            & gm["hgnc_num"].notna()
            & ~gm["hgnc_num"].str.lower().isin(["", "nan", "none"])
        )
        uniq  = sorted(set(gm.loc[valid_mask, "hgnc_num"]),
                       key=lambda x: int(x) if x.isdigit() else 0)
        vocab: dict = {"<pad>": 0, "<cls>": 1}
        for hid in uniq:
            vocab[hid] = len(vocab)

        sym_map = (
            gm.loc[valid_mask, ["hgnc_num", "gene_symbol"]]
            .drop_duplicates("hgnc_num")
            .set_index("hgnc_num")["gene_symbol"]
            .to_dict()
        )

        (out / "gene_vocab.json").write_text(json.dumps(vocab, indent=2))
        conn = {"<pad>": "<pad>", "<cls>": "<cls>"}
        conn.update({hid: sym_map.get(hid, "") for hid in uniq})
        (out / "conn_id_to_hgnc.json").write_text(json.dumps(conn, indent=2))

        # ── Assign hgnc_token ─────────────────────────────────────────────
        gm["hgnc_token"] = gm["hgnc_num"].map(vocab).fillna(0).astype(int)
        gm.loc[~gm["is_hvg"], "hgnc_token"] = 0

        gm.to_parquet(gene_meta_path(intermediate_dir), index=False)
        gm.to_csv(out / "gene_mapping_table.tsv", sep="\t", index=False)

        mapped_n = int(valid_mask.sum())
        return json.dumps({
            "status":     "ok",
            "vocab_size": len(vocab),
            "n_genes":    len(gm),
            "in_vocab":   mapped_n,
            "unmapped":   len(gm) - mapped_n,
            "note":       "only is_hvg=True genes included in vocab",
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
