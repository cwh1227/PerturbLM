import torch
import torch.nn as nn

from genetoken_embedding import GeneEmbedding
from expression_embedding import ExpressionEmbedding
from condition_token_embedding import ConditionEmbedding


# preprocessing/
# ├── genetoken_embedding.py
# ├── expression_embedding.py
# ├── condition_token_embedding.py
# └── total_multiomic_embedding.py


class CellEmbedding(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_conditions: int,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.emb_g = GeneEmbedding(n_genes, d_model)
        self.emb_x = ExpressionEmbedding(d_model)
        self.emb_c = ConditionEmbedding(n_conditions, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, t_g, x, t_c):
        """
        t_g: (B, M) gene tokens
        x:   (B, M) binned expression values
        t_c: (B, M) or (B, 1) condition tokens
        """
        h = (
            self.emb_g(t_g)
            + self.emb_x(x)
            + self.emb_c(t_c)
        )

        return self.dropout(h)  # (B, M, D)


