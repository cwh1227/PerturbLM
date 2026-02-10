import torch

def sample_mask_positions(
    input_ids: torch.Tensor,
    pad_token_id: int,
    cls_index: int = 0,
    mask_ratio: float = 0.15
):
    """
    input_ids: [B, L]  (binned gene tokens, with CLS + PAD)
    returns:
      mask_positions: BoolTensor [B, L]
    """

    B, L = input_ids.shape
    device = input_ids.device

    # candidate positions: not PAD, not CLS
    valid = (input_ids != pad_token_id)
    valid[:, cls_index] = False  # never mask CLS

    # sample Bernoulli mask
    rand = torch.rand((B, L), device=device)
    mask_positions = (rand < mask_ratio) & valid

    return mask_positions


def apply_mask_to_input(
    input_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_token_id: int
):
    """
    Replace selected positions with MASK token.
    """
    masked_input = input_ids.clone()
    masked_input[mask_positions] = mask_token_id
    return masked_input




def build_attention_mask(input_ids, pad_token_id):
    """
    Returns additive attention mask for transformer:
    shape: [B, 1, 1, L]
    """
    mask = (input_ids == pad_token_id).float()  # 1 for PAD
    mask = mask.unsqueeze(1).unsqueeze(2)
    return mask * -10000.0
