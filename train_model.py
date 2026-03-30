import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import LSTMModel

# Load data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Dataset & Loader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ✅ Model
model = LSTMModel(num_classes=4)

# Loss & Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.train()

# Training Loop
epochs = 25

for epoch in range(epochs):
    total_loss = 0

    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/lstm_model.pth")

print("Model trained and saved ✅")