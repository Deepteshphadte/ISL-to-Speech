import torch
import numpy as np

from models.lstm_model import LSTMModel

MODEL_PATH = "models/lstm_model.pth"
LABELS_PATH = "data/labels.npy"

labels = np.load(
    LABELS_PATH,
    allow_pickle=True
)

num_classes = len(labels)

model = LSTMModel(
    input_size=126,
    hidden_size=256,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("✅ LSTM Model Loaded")