import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=126,
        hidden_size=256,
        num_classes=4,
        num_layers=2,
        dropout=0.4
    ):

        super(LSTMModel, self).__init__()

        # ----------------------------
        # LSTM
        # ----------------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ----------------------------
        # Transformer Encoder
        # ----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ----------------------------
        # Classifier
        # ----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # ----------------------------
        # LSTM
        # Output shape:
        # (batch, 30, 256)
        # ----------------------------
        lstm_out, _ = self.lstm(x)

        # ----------------------------
        # Transformer
        # ----------------------------
        transformer_out = self.transformer(lstm_out)

        # ----------------------------
        # Mean Pooling
        # ----------------------------
        features = transformer_out.mean(dim=1)

        # ----------------------------
        # Classification
        # ----------------------------
        output = self.classifier(features)

        return output