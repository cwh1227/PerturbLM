import torch
import torch.nn as nn


class ExpressionEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(1, d_model)

    def forward(self, x):
        """
        x: (B, M)  binned expression values for M genes
        """
        return self.fc(x.unsqueeze(-1))  
