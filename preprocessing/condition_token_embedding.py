import torch
import torch.nn as nn


class ConditionEmbedding(nn.Module):
    def __init__(self, n_conditions: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_conditions, d_model)

    def forward(self, t_c):
        """
        t_c: (B, M) or (B, 1) condition token ids
        """
        return self.emb(t_c)  # broadcast later if needed
