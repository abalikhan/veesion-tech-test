import torch
import torch.nn as nn

def init_weights(module):
    """
    Initialize only trainable parameters in a model.
    """
    # 1) Linear layers
    if isinstance(module, nn.Linear) and module.weight.requires_grad:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and module.bias.requires_grad:
            nn.init.zeros_(module.bias)

    # 2) Multi-head attention
    elif isinstance(module, nn.MultiheadAttention):
        # in-proj weight q/k/v
        if module.in_proj_weight.requires_grad:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None and module.in_proj_bias.requires_grad:
            nn.init.zeros_(module.in_proj_bias)
        # out-proj
        proj = module.out_proj
        if proj.weight.requires_grad:
            nn.init.xavier_uniform_(proj.weight)
        if proj.bias is not None and proj.bias.requires_grad:
            nn.init.zeros_(proj.bias)

    # 3) LayerNorm
    elif isinstance(module, nn.LayerNorm):
        if module.weight.requires_grad:
            nn.init.ones_(module.weight)
        if module.bias.requires_grad:
            nn.init.zeros_(module.bias)

    # 4) LSTM
    elif isinstance(module, nn.LSTM):
        # input-hidden weights & biases
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)