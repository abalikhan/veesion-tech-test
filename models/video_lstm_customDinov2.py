import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor

class AdapterBlock(nn.Module):
    """Custom Bottleneck adapter (with residual)"""
    def __init__(self, hidden_size: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(adapter_dim, hidden_size)
    def forward(self, x):
        # x: (B, hidden)
        return x + self.up(self.act(self.down(x)))

class DinoV2CustomAdapter(nn.Module):
    """
    Customaized Dinov2 model with adapters injected at specific layers
    Args:
      pretrained: HF model identifier
      adapter_dim: hidden size for each adapter
      adapter_layers: list or range of the number of adapter layers for each block
                      (e.g. [0,3,11] or list(range(12)) for all 12 blocks)
    """
    def __init__(
        self,
        pretrained: str = "facebook/dinov2-base",
        adapter_dim: int = 64,
        adapter_layers: list[int] | None = None
    ):
        super().__init__()
        # image processor
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )
        # frozen DINOv2 backbone
        backbone = Dinov2Model.from_pretrained(
            pretrained, trust_remote_code=True
        )
        for p in backbone.parameters():
            p.requires_grad = False
        
        # From HF we know the layers names:
        #   backbone.embeddings             → patch+cls+pos embeddings
        #   backbone.encoder.layer          → ModuleList of Transformer blocks
        #   backbone.encoder.layernorm      → final layer norm
            
        self.embeddings = backbone.embeddings
        self.blocks     = backbone.encoder.layer
        self.norm       = backbone.encoder.layernorm
        
        hidden = backbone.config.hidden_size
        N = len(self.blocks)
        # default: all blocks
        if adapter_layers is None:
            adapter_layers = list(range(N))
        # sanity check
        for i in adapter_layers:
            if not (0 <= i < N):
                raise ValueError(f"Invalid block index {i}, must be in [0..{N-1}]")
        self.adapter_layers = adapter_layers
        
        # one AdapterBlock per chosen layer
        self.adapters = nn.ModuleDict({
            str(i): AdapterBlock(hidden, adapter_dim)
            for i in adapter_layers
        })

    def forward(self, pixel_values: torch.Tensor):
        """
        pixel_values: (B, C, H, W)
        returns:      (B, hidden_size)  the final CLS embedding
        """
        # embed patches + cls + pos
        x = self.embeddings(pixel_values)            # (B, T+1, hidden)
        # run through Transformer block manually
        for idx, blk in enumerate(self.blocks):
            x = blk(x)[0]                            # blk returns (hidden, ...)
            # if this block has an adapter, apply to the CLS token only
            if str(idx) in self.adapters:
                cls = x[:, 0, :]                     # (B, hidden)
                cls = self.adapters[str(idx)](cls)   # (B, hidden)
                x[:, 0, :] = cls
        # final layer norm
        x = self.norm(x)                             # (B, T+1, hidden)
        # return CLS
        return x[:, 0, :]