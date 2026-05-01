import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=4, num_layers=2, dropout=0.4):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = hn[-1]                  # take last layer's hidden state
        return self.classifier(out)