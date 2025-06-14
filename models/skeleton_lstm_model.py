import torch
import torch.nn as nn


# Model to take skeleton keypoints as inputs and model time with lstm for action classification
import torch
import torch.nn as nn

class SkeletonLSTMClassifier(nn.Module):
    def __init__(self, 
                 input_dim=258,         # Number of features per frame
                 hidden_dim=128,        # LSTM hidden dimension
                 num_layers=2,           # Number of LSTM layers
                 num_classes=5,          # Number of output classes
                 dropout=0.3):           # Dropout between LSTM layers 
        super(SkeletonLSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        self.lstm.flatten_parameters()  # Immproves efficiency when using a dataloader with number_of_workers > 0
        _, (hn, _) = self.lstm(x)       # hn: (num_layers * num_directions, B, H)

                    
        out = self.classifier(hn[-1]) # Take the last layer's hidden state (B, H)
        return out
