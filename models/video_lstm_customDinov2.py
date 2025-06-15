import torch
import torch.nn as nn
from models.dino_variable_adapters import DinoV2VariableAdapters
from utils.feats_util import extract_frame_cls_features
from utils.init_weights_utils import init_weights

class VideoLSTMCustomDinov2(nn.Module):
    def __init__(self,
                dinov2_ckpt_path: str,
                num_classes: int,
                adapter_dim: int = 64,
                adapter_layers: list[int] = [0, 5, 11],
                lstm_hidden: int = 256,
                lstm_layers: int = 1,
                dropout: float = 0.3,
                device='cpu'):
        super().__init__()
        # Load our pretrained Custom DinoV2
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

        #LSTM for sequence modelling
        self.lstm = nn.LSTM(
                        input_size=feature_dim,
                        hidden_size=lstm_hidden,
                        num_layers=lstm_layers,
                        batch_first=True,
                        dropout=dropout if lstm_layers > 1 else 0.0
                        )
        
        # classification head
        '''
        Dropout is used here for regularization given the small dataset and high risk of overfitting.
        Similar to task1 LayerNorm could be added if the LSTM output distribution is unstable, 
        but with strong encoders (DINOv2) its rarely critical.
        '''
        self.classifier_head = nn.Sequential(
                                            nn.Dropout(dropout),
                                            nn.Linear(lstm_hidden, num_classes)
                                            )

        # initialize parameters 
        self.apply(init_weights)
        
    def forward(self, pixel_values):
        """
        pixel_seq: Tensor [B, T, C, H, W]
        Returns: logits [B, num_classes]
        """

        # extract features using our helper function
        feats = extract_frame_cls_features(
                                        self.backbone,
                                        pixel_values) #[B, T, feature_dim]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(feats)    # [B, T, hidden]
        
        # regularize and classification head
        logits = self.classifier_head(lstm_out[:, -1, :]) # Take last time step [B, hidden], thus logits will be [B, num_classes]
        return logits
