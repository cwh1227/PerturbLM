import torch
import torch.nn as nn

from preprocessing.total_multiomic_embedding import CellEmbedding
from .transformer_blocks import TransformerEncoderBlock



class DoubleTransformerPerturbLM(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_conditions: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers_gene: int = 4,
        n_layers_perturb: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # ===== Input embedding =====
        self.embedding = CellEmbedding(
            n_genes=n_genes,
            n_conditions=n_conditions,
            d_model=d_model,
            dropout=dropout
        )

        # ===== Transformer A: Gene encoder =====
        self.gene_encoder = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers_gene)
        ])

        # ===== Transformer B: Perturb encoder =====
        self.perturb_encoder = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers_perturb)
        ])

        # ===== Output head =====
        self.head = nn.Linear(d_model, 1)  # predict expression per gene

    def forward(self, t_g, x, t_c):
        """
        t_g: (B, M) gene token ids
        x:   (B, M) binned expression values
        t_c: (B, M) or (B, 1) condition tokens
        """

        # ----- Embedding -----
        h = self.embedding(t_g, x, t_c)  # (B, M, D)

        # ----- Transformer A: gene–gene modeling -----
        for layer in self.gene_encoder:
            h = layer(h)

        # ----- Transformer B: perturbation refinement -----
        for layer in self.perturb_encoder:
            h = layer(h)

        # ----- Prediction -----
        y_hat = self.head(h).squeeze(-1)  # (B, M)

        return y_hat
