import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.lstm_model import LSTMModel

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "data"
MODEL_PATH = "models/lstm_model.pth"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))
labels = np.load(
    os.path.join(DATA_PATH, "labels.npy"),
    allow_pickle=True
)

print(f"X Shape : {X.shape}")
print(f"y Shape : {y.shape}")

# --------------------------------------------------
# Convert to Tensor
# --------------------------------------------------

X_tensor = torch.tensor(X, dtype=torch.float32)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

num_classes = len(labels)

model = LSTMModel(
    input_size=126,
    hidden_size=256,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.4
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

model.eval()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    outputs = model(X_tensor)

    preds = torch.argmax(outputs, dim=1)

y_pred = preds.numpy()

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(figsize=(14, 14))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n✅ Confusion matrix saved as:")
print("confusion_matrix.png")