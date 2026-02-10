# ============================================================
# XPertNet v2 (clean / extensible)
# - Control path: self-attention only
# - Treatment path: cross-attn at FIRST and LAST layer (cell <-> drug),
#                   self-attn in the middle
# - Better masking hooks + buffer management + clear structure
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from models.model_utils import (
    Encoder,          # self-attn block: (x, attn_mask, sparse_flag, output_attention) -> (x, attn)
    crossEncoder,     # cross-attn block: (cell_x, drug_x, drug_mask, cell_mask, sparse_flag, output_attention)
                      #                -> (cell_x, drug_x, attn)
    cell_Embeddings,
    unimol_Embeddings
)

# -------------------------
# Utils
# -------------------------

def get_unimol_drug_feat(input_features: torch.Tensor):
    """
    input_features: [B, A, F]
      col0: atom_mask_raw (1=valid atom, 0=pad)
      col1: atom_symbol (int)
      col2+: atom_feat (float)
    returns:
      atom_feat: [B, A, F-2]
      atom_symbol: [B, A]
      attn_mask: [B, 1, 1, A] with 0 on valid, -10000 on pad (additive mask)
    """
    atom_mask_raw = input_features[:, :, 0].long()
    atom_symbol   = input_features[:, :, 1].long()
    atom_feat     = input_features[:, :, 2:].float()

    attn_mask = atom_mask_raw.unsqueeze(1).unsqueeze(2).float()  # [B,1,1,A]
    attn_mask = (1.0 - attn_mask) * -10000.0

    return atom_feat, atom_symbol, attn_mask


def make_full_attention_mask(batch_size: int, seq_len: int, device: torch.device):
    """
    If you have no padding and want explicit masks, you can use this (all valid => zeros).
    Returns additive mask [B, 1, 1, L] filled with 0.0
    """
    return torch.zeros((batch_size, 1, 1, seq_len), device=device, dtype=torch.float32)


# -------------------------
# Attention stacks
# -------------------------

class SelfAttnStack(nn.Module):
    """
    Pure self-attention stack over cell tokens.
    Extensible: add norms, residual scaling, etc. inside Encoder if needed.
    """
    def __init__(self, hidden_size, intermediate_size, n_heads,
                 attn_drop, hidden_drop, topk_cell, n_layers, sparse_flag, logger=None):
        super().__init__()
        self.sparse_flag = sparse_flag
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            Encoder(hidden_size, intermediate_size, n_heads, attn_drop, hidden_drop, topk_cell)
            for _ in range(n_layers)
        ])
        if logger:
            logger.info(f"[SelfAttnStack] layers={n_layers}, sparse={sparse_flag}")

    def forward(self, x, x_mask=None, output_attention=False):
        attn_dict = {} if output_attention else None
        for i, block in enumerate(self.layers):
            x, attn = block(x, x_mask, self.sparse_flag, output_attention)
            if output_attention:
                attn_dict[f"SA_{i}"] = attn
        return x, attn_dict


class CrossFirstLastStack(nn.Module):
    """
    Treatment path:
      layer0: Cross-Attn (cell <-> drug)
      middle: N self-attn layers (cell only)
      last : Cross-Attn (cell <-> drug)
    Optional hook: inject drug-specific gene embedding between layers.
    """
    def __init__(self, hidden_size, intermediate_size, n_heads,
                 attn_drop, hidden_drop, topk_cell, topk_drug,
                 n_self_layers_mid, sparse_flag, logger=None):
        super().__init__()
        self.sparse_flag = sparse_flag

        self.cross_first = crossEncoder(hidden_size, intermediate_size, n_heads,
                                        attn_drop, hidden_drop, topk_cell, topk_drug)

        self.mid_self = nn.ModuleList([
            Encoder(hidden_size, intermediate_size, n_heads, attn_drop, hidden_drop, topk_cell)
            for _ in range(n_self_layers_mid)
        ])

        self.cross_last = crossEncoder(hidden_size, intermediate_size, n_heads,
                                       attn_drop, hidden_drop, topk_cell, topk_drug)

        if logger:
            logger.info(f"[CrossFirstLastStack] mid_self_layers={n_self_layers_mid}, sparse={sparse_flag}")

    def forward(self, cell_x, drug_x, cell_mask=None, drug_mask=None,
                output_attention=False,
                drug_specific_gene_embedding=None,
                lambda_specific=0.0):
        """
        cell_x: [B, Lc, H]
        drug_x: [B, Ld, H]
        masks: additive masks [B,1,1,L]
        drug_specific_gene_embedding: [B, Lc, H] (must align with cell tokens!)
        """
        attn_dict = {} if output_attention else None

        # 1) first cross-attn
        cell_x, drug_x, attn = self.cross_first(cell_x, drug_x, drug_mask, cell_mask,
                                                self.sparse_flag, output_attention)
        if output_attention:
            attn_dict["CA_first"] = attn

        # optional injection after first cross (common pattern)
        if (drug_specific_gene_embedding is not None) and (lambda_specific != 0.0):
            cell_x = cell_x + lambda_specific * drug_specific_gene_embedding

        # 2) mid self-attn
        for i, block in enumerate(self.mid_self):
            cell_x, attn = block(cell_x, cell_mask, self.sparse_flag, output_attention)
            if output_attention:
                attn_dict[f"SA_mid_{i}"] = attn

            # optional injection between mid layers (if you want)
            if (drug_specific_gene_embedding is not None) and (lambda_specific != 0.0):
                cell_x = cell_x + lambda_specific * drug_specific_gene_embedding

        # 3) last cross-attn
        cell_x, drug_x, attn = self.cross_last(cell_x, drug_x, drug_mask, cell_mask,
                                               self.sparse_flag, output_attention)
        if output_attention:
            attn_dict["CA_last"] = attn

        return cell_x, drug_x, attn_dict


# -------------------------
# Main model
# -------------------------

class XPertNetV2(nn.Module):
    """
    Drop-in style replacement for your XPertNet, but with clearer structure:
      - control_encoder: self-attn stack
      - treatment_encoder: cross-first-last stack
    """

    def __init__(self, args, config, device, logger):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device
        self.logger = logger

        # ===== dataset params =====
        max_gene_length = config["dataset"]["gene_num"]
        exp_vocab_size  = config["dataset"]["n_bins"]

        atom_num      = config["dataset"]["atom_num"]
        max_atom_size = config["dataset"]["max_atom_size"]

        # ===== model params =====
        hidden_size = config["model"]["ATTN"]["hidden_size"]
        self.hidden_size = hidden_size

        n_heads    = config["model"]["ATTN"]["n_heads"]
        topk_cell  = config["model"]["ATTN"]["topk_cell"]
        topk_drug  = config["model"]["ATTN"]["topk_drug"]

        attn_drop  = config["model"]["ATTN"]["attention_probs_dropout_prob"]
        hid_drop   = config["model"]["ATTN"]["hidden_dropout_prob"]

        cell_in_drop = config["model"]["ATTN"]["cell_input_hidden_dropout_prob"]
        drug_in_drop = config["model"]["ATTN"]["drug_input_hidden_dropout_prob"]

        self.sparse_flag = config["model"]["ATTN"]["sparse_flag"]
        logger.info(f"[XPertNetV2] sparse_attn={self.sparse_flag}")

        intermediate_size = hidden_size * 2

        # ===== cell embedding =====
        logger.info("[XPertNetV2] cell embedding: ppi_gene_vector + exp embedding")
        ppi_path = config["model"]["ATTN"]["ppi_gene_vector_path"]
        self.cell_emb = cell_Embeddings(
            exp_vocab_size, hidden_size, max_gene_length, cell_in_drop, ppi_path, args
        )

        # ===== drug embedding =====
        if "sdst" in args.dataset:
            self.drug_emb = unimol_Embeddings(atom_num, hidden_size, max_atom_size, drug_in_drop, args)
        elif "mdmt" in args.dataset:
            logger.info("[XPertNetV2] using dose/time embedding")
            ds_key = args.dataset.split("_")[0]
            pert_dose_emb = nn.Embedding(config["dataset"][ds_key]["num_pert_dose"], hidden_size)
            pert_time_emb = nn.Embedding(config["dataset"][ds_key]["num_pert_time"], hidden_size)
            self.drug_emb = unimol_Embeddings(atom_num, hidden_size, max_atom_size, drug_in_drop, args,
                                             pert_dose_emb, pert_time_emb)
        else:
            raise ValueError(f"Unknown dataset flavor for drug embedding: {args.dataset}")

        # ===== HG embeddings as buffers (recommended) =====
        drug_hg_path = config["model"]["HG"]["drug_hg_pretrained_embed_path"]
        drug_hg = torch.from_numpy(np.load(drug_hg_path, allow_pickle=False)).float()
        self.register_buffer("drug_HG_embed", drug_hg, persistent=True)

        # ===== optional drug-specific gene embedding =====
        self.lambda_specific = float(getattr(args, "lambda_specific", 0.0))  # easy knob from args
        self.use_specific = (args.pretrained_mode == "specific")
        if self.use_specific:
            spec_path = config["model"]["HG"]["specific_pretrained_embed_path"]
            spec = torch.from_numpy(np.load(spec_path, allow_pickle=False)).float()
            self.register_buffer("drug_specific_gene_embed", spec, persistent=True)

            # transform to align space / stabilize training
            self.transform_specific = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            logger.info(f"[XPertNetV2] specific mode ON, lambda_specific={self.lambda_specific}")
        else:
            logger.info("[XPertNetV2] specific mode OFF")

        # ===== encoders =====
        # Control: pure self-attn
        n_ctl_layers = int(config["model"]["ATTN"].get("ctl_layers", 4))
        self.control_encoder = SelfAttnStack(
            hidden_size, intermediate_size, n_heads,
            attn_drop, hid_drop, topk_cell,
            n_layers=n_ctl_layers,
            sparse_flag=self.sparse_flag,
            logger=logger
        )

        # Treatment: cross first + self(mid) + cross last
        n_mid = int(config["model"]["ATTN"].get("trt_mid_self_layers", 2))
        self.treatment_encoder = CrossFirstLastStack(
            hidden_size, intermediate_size, n_heads,
            attn_drop, hid_drop, topk_cell, topk_drug,
            n_self_layers_mid=n_mid,
            sparse_flag=self.sparse_flag,
            logger=logger
        )

        # ===== heads (token-wise regression) =====
        latent = 64
        self.ctl_fc = nn.Sequential(nn.Linear(hidden_size, latent), nn.ReLU(), nn.Dropout(0.1), nn.Linear(latent, 1))
        self.trt_fc = nn.Sequential(nn.Linear(hidden_size, latent), nn.ReLU(), nn.Dropout(0.1), nn.Linear(latent, 1))
        self.deg_fc = nn.Sequential(nn.Linear(hidden_size, latent), nn.ReLU(), nn.Dropout(0.1), nn.Linear(latent, 1))

        # ===== optional CLS classification =====
        self.include_cell_idx = bool(getattr(args, "include_cell_idx", False))
        if self.include_cell_idx:
            ds_key = args.dataset.split("_")[0]
            num_cell_id = config["dataset"][ds_key]["num_cell_id"]
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.class_fc  = nn.Linear(hidden_size, num_cell_id)
            logger.info(f"[XPertNetV2] include_cell_idx ON, num_cell_id={num_cell_id}")

    # -------------------------
    # forward
    # -------------------------
    def forward(self, data, mode="ST"):
        """
        data tuple (kept compatible with your original):
          trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,
          drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx
        """
        (trt_raw_data, ctl_raw_data,
         trt_raw_data_binned, ctl_raw_data_binned,
         drug_feat, pert_dose_idx, pert_time_idx,
         drug_idx, cell_idx, tissue_idx) = data

        # move essentials to device
        trt_raw_data        = trt_raw_data.to(self.device)
        ctl_raw_data        = ctl_raw_data.to(self.device)
        trt_raw_data_binned = trt_raw_data_binned.to(self.device)
        ctl_raw_data_binned = ctl_raw_data_binned.to(self.device)

        drug_feat     = drug_feat.to(self.device)
        pert_dose_idx = pert_dose_idx.to(self.device)
        pert_time_idx = pert_time_idx.to(self.device)
        drug_idx      = drug_idx.to(self.device)
        cell_idx      = cell_idx.to(self.device)
        tissue_idx    = tissue_idx.to(self.device)

        B = cell_idx.shape[0]
        cell_class_true = cell_idx

        # ---- drug feature parse ----
        if self.args.drug_feat == "unimol":
            drug_unimol_embed, drug_atom_symbols, drug_mask = get_unimol_drug_feat(drug_feat)
        else:
            drug_unimol_embed = drug_feat
            drug_atom_symbols = None
            drug_mask = None

        # ---- build drug token embedding ----
        drug_hg = self.drug_HG_embed[drug_idx].to(self.device)  # buffer slice -> on correct device
        drug_tokens = self.drug_emb(drug_unimol_embed, drug_hg, drug_atom_symbols, pert_dose_idx, pert_time_idx)

        # ---- build cell token embedding from CONTROL binned ----
        cell_tokens = self.cell_emb(ctl_raw_data_binned)  # [B, Lg, H]

        # ---- optional CLS token ----
        if self.include_cell_idx:
            cls = self.cls_token.expand(B, -1, -1)
            cell_tokens = torch.cat([cls, cell_tokens], dim=1)  # [B, 1+Lg, H]

        # ---- masks (optional) ----
        # If you truly have fixed length with no padding, you can keep cell_mask=None.
        # If you want explicit masks, uncomment:
        # cell_mask = make_full_attention_mask(B, cell_tokens.shape[1], self.device)
        cell_mask = None

        # ---- optional drug-specific gene embedding aligned to cell tokens ----
        specific_embed = None
        if self.use_specific:
            # Expect shape: [num_drug, Lg, H] OR [num_drug, 1+Lg, H] depending on how you stored it.
            # We'll assume stored as [num_drug, Lg, H]. If you have CLS, pad a zero at front.
            spec = self.drug_specific_gene_embed[drug_idx].to(self.device)  # [B, Lg, H]
            spec = self.transform_specific(spec)  # [B, Lg, H]

            if self.include_cell_idx:
                # prepend zeros for CLS position
                zero = torch.zeros((B, 1, self.hidden_size), device=self.device, dtype=spec.dtype)
                spec = torch.cat([zero, spec], dim=1)  # [B, 1+Lg, H]
            specific_embed = spec

        # =========================
        # 1) CONTROL path (self-attn only)
        # =========================
        ctl_cell_ctx, ctl_attn = self.control_encoder(
            cell_tokens, x_mask=cell_mask, output_attention=getattr(self.args, "output_attention", False)
        )

        # =========================
        # 2) TREATMENT path (cross first + self mid + cross last)
        # =========================
        trt_cell_ctx, drug_tokens_out, trt_attn = self.treatment_encoder(
            cell_tokens, drug_tokens,
            cell_mask=cell_mask, drug_mask=drug_mask,
            output_attention=getattr(self.args, "output_attention", False),
            drug_specific_gene_embedding=specific_embed,
            lambda_specific=self.lambda_specific
        )

        # ---- if CLS enabled: split out CLS and gene tokens ----
        if self.include_cell_idx:
            ctl_cls = ctl_cell_ctx[:, 0, :]
            trt_cls = trt_cell_ctx[:, 0, :]
            ctl_gene_ctx = ctl_cell_ctx[:, 1:, :]
            trt_gene_ctx = trt_cell_ctx[:, 1:, :]

            cell_class_predict = (self.class_fc(ctl_cls), self.class_fc(trt_cls))
            cls_embed = torch.stack([trt_cls, ctl_cls], dim=1)  # [B,2,H]
        else:
            ctl_gene_ctx = ctl_cell_ctx
            trt_gene_ctx = trt_cell_ctx
            cell_class_predict = None
            cls_embed = None

        # ---- heads ----
        trt_output = self.trt_fc(trt_gene_ctx).squeeze(-1)                 # [B, Lg]
        ctl_output = self.ctl_fc(ctl_gene_ctx).squeeze(-1)                 # [B, Lg]
        deg_output = self.deg_fc(trt_gene_ctx - ctl_gene_ctx).squeeze(-1)  # [B, Lg]

        attention_dict = (trt_attn, ctl_attn)

        if getattr(self.args, "mode", "") == "infer":
            return (trt_output, ctl_output, deg_output,
                    trt_raw_data, ctl_raw_data,
                    attention_dict,
                    cell_class_true, cell_class_predict,
                    cls_embed)
        else:
            return (trt_output, ctl_output, deg_output,
                    trt_raw_data, ctl_raw_data,
                    attention_dict,
                    cell_class_true, cell_class_predict)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
