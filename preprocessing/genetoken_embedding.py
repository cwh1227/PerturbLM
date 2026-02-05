import torch
import torch.nn as nn


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_genes, d_model)

    def forward(self, t_g):
        """
        t_g: (B, M)  gene token ids ，B=size of batch,M=number of genes selected for this cell
        returns: (B, M, D) gene embeddings
        """
        return self.emb(t_g)  

