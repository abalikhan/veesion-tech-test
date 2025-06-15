import torch

def extract_frame_cls_features(backbone, pixel_sequences):
    """
    Given a vision backbone that accepts `pixel_values=…` and returns a
    ModelOutput with `last_hidden_state` of shape [batch, seq_len+1, hidden],
    this will:

      1. Flatten [B, T, C, H, W] --> [B*T, C, H, W]
      3. Run the backbone and grab the CLS token [:, 0, :]
      4. Reshape back to [B, T, hidden]

    Args:
      backbone: our custom pretrained backbone
      pixel_sequences: Tensor [B, T, C, H, W]
    Returns:
      feats: Tensor [B, T, hidden_dim]
    """
    B, T, C, H, W = pixel_sequences.shape
    frames = pixel_sequences.view(B * T, C, H, W)
    with torch.no_grad():
        out = backbone(pixel_values=frames)
    
    '''
    we need to check where the output coming from. either transformer or lstm
    '''
    # Case 1: HuggingFace ModelOutput
    if hasattr(out, "last_hidden_state"):
        # Grab the CLS token embedding
        cls_feats = out.last_hidden_state[:, 0, :]    # [B*T, hidden]
    else:
      
        # out: [B*T, hidden]
        cls_feats = out

    # Reshape back to [B, T, hidden]
    return cls_feats.view(B, T, -1)