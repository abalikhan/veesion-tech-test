import torch
import torch.nn as nn
from models.dino_variable_adapters import DinoV2VariableAdapters
from utils.feats_util import extract_frame_cls_features
import math 
from utils.init_weights_utils import init_weights

class VideoTransformerDinov2(nn.Module):
    def __init__(self,
                dinov2_ckpt_path: str,
                num_classes: int,
                adapter_dim: int = 64,
                adapter_layers: list[int] = [0, 5, 11],
                num_heads: int = 256,
                num_layers: int = 1,
                use_cls_token = True,  # if False, use mean pooling
                dropout: float = 0.3,
                device='cpu'
                ):
        
        super().__init__()
        # Load our pretrained Custom DinoV2

        self.use_cls_token = use_cls_token
        self.backbone = DinoV2VariableAdapters(
                                            adapter_dim=adapter_dim,
                                            adapter_layers=adapter_layers
                                                )
        if dinov2_ckpt_path:
            state_dict = torch.load(dinov2_ckpt_path, map_location=device, weights_only=True)
            self.backbone.load_state_dict(state_dict)
        # Freeze our pretrained backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # get the last layer feature dimensions from our pretrained model
        feature_dim = self.backbone.norm.normalized_shape[0] # usually 768 for Dinov2

        #Transformer Encoder for Sequence modelling
        transformer_encoder = nn.TransformerEncoderLayer(
                                                           d_model=feature_dim,
                                                           nhead=num_heads,
                                                           dim_feedforward=(feature_dim*4), # as per the paper "Attention is all you need"
                                                           dropout=dropout,
                                                           batch_first=True
                                                        )
        
        self.transformer = nn.TransformerEncoder(
                                                   encoder_layer=transformer_encoder,
                                                   num_layers=num_layers,

                                                )
        # An option to use learnable CLS token or just avg pool the features.
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # classification head
        '''
        We opt layernorm here as its essential for stability and dropout already exists in transformer layer
        adding it again will be redundant.
        '''
        self.classifier_head = nn.Sequential(
                                            nn.LayerNorm(feature_dim),
                                            nn.Linear(feature_dim, num_classes)
                                            )

        # initialize parameters 
        self.apply(init_weights)

    def _get_positional_encoding(self, seq_len: int, dim: int, device: torch.device):
        """
        Create a [seq_len, dim] sinusoidal positional encoding.
        """
        pe = torch.zeros(seq_len, dim, device=device)
        position = torch.arange(0, seq_len, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [seq_len, dim]
    
    def forward(self, pixel_values):
        """
        pixel_seq: Tensor [B, T, C, H, W]
        Returns: logits [B, num_classes]
        """
        B, T, _, _, _ = pixel_values.shape   #Since we are not using C, H, W so best is to avoid extra parameters initialization
        # extract features using our helper function
        feats = extract_frame_cls_features(
                                        self.backbone,
                                        pixel_values)  # [B, T, feature_dim]

        # Prepend a learnable CLS token if its True
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, feature_dim)
            feats = torch.cat([cls_tokens, feats], dim=1)  # (B, T+1, feature_dim)
            seq_len = T + 1                                # 
        else:
            seq_len = T
        
        # adding the positional embedding to inform our transformer about frame sequences
        pe = self._get_positional_encoding(seq_len, feats.size(-1), feats.device)
        feats = feats + pe.unsqueeze(0)  # broadcast to [B, seq_len, D]

        
        # Temporal modelling with Transformer encoder
        out = self.transformer(feats)    # [B, seq_len, D]

        # use CLS token or mean over time
        if self.use_cls_token:
            video_feat = out[:, 0, :]    # [B, D]
        else:
            video_feat = out.mean(dim=1) # [B, D]

        # Normalize & classify
        logits = self.classifier_head(video_feat)
        return logits


