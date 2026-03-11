"""
Unified tokenize + shard tool.

Reads from standardized intermediate files:
    intermediate/cell_meta.parquet   — QC-filtered cells + metadata
    intermediate/gene_meta.parquet   — HVG-selected, HGNC-mapped genes
    intermediate/data_source.json    — {format, path, ...} for expression data

All datasets produce identical .npz shards:

    sample_id  U       [B]        cell / sample identifier
    gene_ids   int32   [B, K]     HGNC numeric ID token  (0 = pad)
    bin_ids    int16   [B, K]     rank-based expression bin  (0 = pad)
    lengths    int16   [B]        effective gene count per cell  (≤ topk)
    text_data  U       [B, 15]    tissue / region / sampleType / compartment /
                                   cellType1 / cellType2 / majorCluster / subCluster /
                                   sex / age / perturbType / disease / drug / dose / time
    text_cols  U       [15]       column name labels (always the same 15)
    domain_id  int32   [B]        integer domain / batch label

Gene encoding:  original identifier  →  HGNC numeric ID token
                (hgnc_token column in gene_meta.parquet, set by map_to_hgnc)

Expression format dispatch is internal — the agent calls one tool regardless of format.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from Agent.state import cell_meta_path, data_source_path, gene_meta_path

_TEXT_COLS = [
    "tissue", "region", "sampleType", "compartment",
    "cellType1", "cellType2", "majorCluster", "subCluster",
    "sex", "age", "perturbType", "disease", "drug", "dose", "time",
]


def _text_row(row: pd.Series) -> list[str]:
    return [str(row.get(c, "")) for c in _TEXT_COLS]


def _rank_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    L = len(values)
    return (n_bins - (np.arange(L, dtype=np.int64) * n_bins) // L).astype(np.int16)


def _encode_cell(expr: np.ndarray, gene_tok: np.ndarray,
                 topk: int, n_bins: int, use_abs: bool
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    if use_abs:
        expr = np.abs(expr)
    tokens = gene_tok * (expr != 0).astype(np.int32)
    order  = np.argsort(expr)[::-1]
    tok_s  = tokens[order]
    val_s  = expr[order]
    valid  = tok_s > 0
    tok_v  = tok_s[valid][:topk]
    val_v  = val_s[valid][:topk]
    L      = len(tok_v)
    gene_row = np.zeros(topk, dtype=np.int32)
    bin_row  = np.zeros(topk, dtype=np.int16)
    if L > 0:
        gene_row[:L] = tok_v
        bin_row[:L]  = _rank_bin(val_v, n_bins)
    return gene_row, bin_row, L


def _write_shard(shard_dir: Path, idx: int, buf: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_dir / f"shard_{idx:05d}.npz",
        sample_id=np.array(buf["sample_id"], dtype=object),
        gene_ids =np.stack(buf["gene_ids"]),
        bin_ids  =np.stack(buf["bin_ids"]),
        lengths  =np.array(buf["lengths"],   dtype=np.int16),
        domain_id=np.array(buf["domain_id"], dtype=np.int32),
        text_data=np.array(buf["text_data"], dtype=object),
        text_cols=np.array(_TEXT_COLS,       dtype=object),
    )


def _empty_buf() -> dict:
    return {k: [] for k in
            ("sample_id", "gene_ids", "bin_ids", "lengths", "domain_id", "text_data")}


# ---------------------------------------------------------------------------
# Single unified tokenize tool
# ---------------------------------------------------------------------------

@tool
def tokenize(
    out_dir:           str,
    topk:              int  = 1024,
    n_bins:            int  = 32,
    shard_size:        int  = 50000,
    use_abs:           bool = True,
    domain_key:        str  = "datasetID",
    # h5ad-specific
    chunk_size:        int  = 50000,
    # gctx-specific
    extra:             int  = 128,
    chunk_rows:        int  = 20000,
    # hf-specific
    batch_size:        int  = 4096,
    expression_subset: str  = "expression_data",
    intermediate_dir:  str  = "",
) -> str:
    """
    Tokenize expression data and write compressed .npz shards.

    Reads from intermediate parquet files (written by read_*, qc_filter_cells,
    select_hvg, map_to_hgnc) — no format-specific parameters needed.

    Output per shard (same schema for all datasets):
        sample_id  U      [B]      cell / sample identifier
        gene_ids   int32  [B, K]   HGNC numeric ID token (0 = pad)
        bin_ids    int16  [B, K]   rank-based expression bin (0 = pad)
        lengths    int16  [B]      effective gene count per cell
        text_data  U      [B, 15]  tissue / region / sampleType / compartment /
                                    cellType1 / cellType2 / majorCluster / subCluster /
                                    sex / age / perturbType / disease / drug / dose / time
        text_cols  U      [15]     column name labels
        domain_id  int32  [B]      integer domain label (from domain_key column)

    Expression format is auto-detected from intermediate/data_source.json.
    chunk_size (h5ad), extra+chunk_rows (gctx), batch_size (hf) are used
    only for the matching format.

    Writes shards to out_dir/shards/  and domain_map.json to out_dir/.
    Returns JSON with n_shards, n_cells, and domain count.
    """
    try:
        cell_meta = pd.read_parquet(cell_meta_path(intermediate_dir))
        gene_meta = pd.read_parquet(gene_meta_path(intermediate_dir))
        data_src  = json.loads(data_source_path(intermediate_dir).read_text())
        fmt       = data_src.get("format", "")

        out       = Path(out_dir)
        shard_dir = out / "shards"

        # Domain map
        domain_map: dict[str, int] = {}
        if domain_key and domain_key in cell_meta.columns:
            for v in sorted(cell_meta[domain_key].astype(str).unique()):
                domain_map[v] = len(domain_map)

        def _dom(row: pd.Series) -> int:
            return domain_map.get(str(row.get(domain_key, "")), 0)

        shard_idx      = 0
        n_cells_written = 0
        buf = _empty_buf()

        def _flush():
            nonlocal shard_idx, buf
            _write_shard(shard_dir, shard_idx, buf)
            shard_idx += 1
            buf = _empty_buf()

        def _emit(sid: str, gene_row, bin_row, length, domain_id, meta_row):
            nonlocal n_cells_written
            if length == 0:
                return
            buf["sample_id"].append(sid)
            buf["gene_ids"].append(gene_row)
            buf["bin_ids"].append(bin_row)
            buf["lengths"].append(np.int16(length))
            buf["domain_id"].append(np.int32(domain_id))
            buf["text_data"].append(_text_row(meta_row))
            n_cells_written += 1
            if len(buf["sample_id"]) >= shard_size:
                _flush()

        # ── h5ad ─────────────────────────────────────────────────────────
        if fmt == "h5ad":
            import anndata as ad
            import scipy.sparse as sp

            adata     = ad.read_h5ad(data_src["path"], backed="r")
            var_names = adata.var_names.tolist()
            gm_lookup = gene_meta.set_index("orig_id")["hgnc_token"].to_dict()
            gene_tok  = np.array([gm_lookup.get(g, 0) for g in var_names], dtype=np.int32)
            kept      = set(cell_meta.index.astype(str))

            for start in range(0, adata.shape[0], chunk_size):
                chunk = adata[start: start + chunk_size]
                X = chunk.X
                if sp.issparse(X):
                    X = X.toarray()
                X = X.astype(np.float32)
                for ci, barcode in enumerate(chunk.obs_names):
                    sid = str(barcode)
                    if sid not in kept:
                        continue
                    meta_row = cell_meta.loc[sid] if sid in cell_meta.index else pd.Series()
                    g, b, L  = _encode_cell(X[ci], gene_tok, topk, n_bins, use_abs)
                    _emit(sid, g, b, L, _dom(meta_row), meta_row)
            adata.file.close()

        # ── gctx ─────────────────────────────────────────────────────────
        elif fmt == "gctx":
            # Build gctx_index → hgnc_token array from gene_meta
            gctx_gm  = gene_meta[gene_meta["gctx_index"] >= 0].copy()
            max_idx  = int(gctx_gm["gctx_index"].max())
            gctx_map = np.zeros(max_idx + 1, dtype=np.int32)
            for _, row in gctx_gm[gctx_gm["hgnc_token"] > 0].iterrows():
                gctx_map[int(row["gctx_index"])] = int(row["hgnc_token"])

            script = Path(__file__).parent.parent.parent / "data_process_L1000_v0.1.py"
            spec   = importlib.util.spec_from_file_location("_l1000_tok", str(script))
            m      = importlib.util.module_from_spec(spec)
            sys.modules["_l1000_tok"] = m
            spec.loader.exec_module(m)

            domain_map = m.step_tokenize_shards(
                gctx_path=Path(data_src["path"]),
                out_dir=out,
                map_g_to_hgnctoken=gctx_map,
                cond_keep=cell_meta,
                topk=topk, n_bins=n_bins, extra=extra,
                chunk_rows=chunk_rows, shard_size=shard_size,
                use_abs=use_abs, text_cols=_TEXT_COLS,
            )
            (out / "domain_map.json").write_text(json.dumps(domain_map, indent=2))
            shards = list(shard_dir.glob("*.npz"))
            return json.dumps({
                "status": "ok", "n_shards": len(shards),
                "domains": len(domain_map), "out_dir": str(out), "format": fmt,
            })

        # ── HuggingFace ───────────────────────────────────────────────────
        elif fmt == "hf":
            # Build orig_token_id → hgnc_token array from gene_meta
            hf_gm   = gene_meta[gene_meta["orig_token_id"] >= 0].copy()
            max_tok = int(hf_gm["orig_token_id"].max())
            orig_to_hgnctoken = np.zeros(max_tok + 1, dtype=np.int32)
            for _, row in hf_gm[hf_gm["hgnc_token"] > 0].iterrows():
                orig_to_hgnctoken[int(row["orig_token_id"])] = int(row["hgnc_token"])

            script = Path(__file__).parent.parent.parent / "data_process_Tahoe.py"
            spec   = importlib.util.spec_from_file_location("_tahoe_tok", str(script))
            m      = importlib.util.module_from_spec(spec)
            sys.modules["_tahoe_tok"] = m
            spec.loader.exec_module(m)

            subset_split = data_src.get("subset_split", "train")
            domain_map   = m.step_tokenize_shards(
                dataset_dir=Path(data_src["path"]),
                out_dir=out,
                obs_keep=cell_meta,
                orig_to_hgnctoken=orig_to_hgnctoken,
                topk=topk, n_bins=n_bins, batch_size=batch_size,
                shard_size=shard_size, use_abs=use_abs,
                text_cols_list=_TEXT_COLS,
                subset_split=subset_split,
                expression_subset=expression_subset,
            )
            (out / "domain_map.json").write_text(json.dumps(domain_map, indent=2))
            shards = list(shard_dir.glob("*.npz"))
            return json.dumps({
                "status": "ok", "n_shards": len(shards),
                "domains": len(domain_map), "out_dir": str(out), "format": fmt,
            })

        else:
            return json.dumps({"error": f"Unknown format '{fmt}' in data_source.json"})

        # Flush remaining h5ad buffer
        if buf["sample_id"]:
            _flush()

        if domain_map:
            (out / "domain_map.json").write_text(json.dumps(domain_map, indent=2))

        shards = list(shard_dir.glob("*.npz"))
        return json.dumps({
            "status":   "ok",
            "n_shards": len(shards),
            "n_cells":  n_cells_written,
            "domains":  len(domain_map),
            "out_dir":  str(out),
            "format":   fmt,
        })
    except Exception:
        return json.dumps({"error": traceback.format_exc()})
