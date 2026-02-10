import torch
import torch.nn as nn


class ExpressionRegressionHead(nn.Module):
    """
    Gene-level regression head.
    Predicts exact expression values for each gene token.

    Input:
      gene_tokens: [B, Lg, H]

    Output:
      expr_pred:  [B, Lg]
    """

    def __init__(self, hidden_size, latent_size=64, dropout=0.1, logger=None):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, 1)
        )

        if logger:
            logger.info(
                f"[CROSSCELL][ExprHead] initialized "
                f"(latent={latent_size}, dropout={dropout})"
            )

    def forward(self, gene_tokens):
        """
        gene_tokens: [B, Lg, H]
        """
        expr = self.proj(gene_tokens).squeeze(-1)  # [B, Lg]
        return expr
