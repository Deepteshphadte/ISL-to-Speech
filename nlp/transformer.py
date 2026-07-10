import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        max_length=50,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embed_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = (
            self.embedding(x)
            + self.position_embedding(positions)
        )

        x = self.transformer(x)

        x = self.fc(x)

        return x