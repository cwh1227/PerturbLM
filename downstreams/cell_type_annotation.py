class CellPooling(nn.Module):
    """
    Pool gene-level tokens into a single cell-level representation.

    Supported modes:
      - 'cls'   : use CLS token (index 0)
      - 'mean'  : mean over gene tokens
      - 'attn'  : learnable attention pooling
    """

    def __init__(self, hidden_size, mode="cls", logger=None):
        super().__init__()
        self.mode = mode

        if mode == "attn":
            self.attn_score = nn.Linear(hidden_size, 1)

        if logger:
            logger.info(f"[CROSSCELL][Pooling] mode = {mode}")

    def forward(self, tokens):
        """
        tokens:
          - if cls:  [B, 1+Lg, H]
          - else:    [B, Lg, H]
        """
        if self.mode == "cls":
            return tokens[:, 0, :]  # [B, H]

        elif self.mode == "mean":
            return tokens.mean(dim=1)  # [B, H]

        elif self.mode == "attn":
            # attention pooling
            scores = self.attn_score(tokens)        # [B, L, 1]
            weights = torch.softmax(scores, dim=1)  # [B, L, 1]
            pooled = (tokens * weights).sum(dim=1)  # [B, H]
            return pooled

        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")
