import torch.nn as nn

class MaskedBinPredictionHead(nn.Module):
    """
    Predict binned expression value for masked gene tokens.
    """

    def __init__(self, hidden_size, vocab_size, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)  # logits over bins
        )

    def forward(self, hidden_states):
        """
        hidden_states: [B, L, H]
        returns logits: [B, L, vocab_size]
        """
        return self.proj(hidden_states)
