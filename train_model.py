import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.lstm_model import LSTMModel

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH     = "data"
MODEL_SAVE    = "models/lstm_model.pth"
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.2
PATIENCE      = 15
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    X = np.load(os.path.join(DATA_PATH, "X.npy"))
    y = np.load(os.path.join(DATA_PATH, "y.npy"))
    print(f"Loaded X: {X.shape}  |  y: {y.shape}")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def train():
    X, y = load_data()

    num_classes = len(torch.unique(y))
    print(f"Number of classes detected: {num_classes}")

    dataset    = TensorDataset(X, y)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}  |  Val samples: {val_size}")

    # ── Model ──
    input_size = X.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=256,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # — Train —
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * X_batch.size(0)
            preds          = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += X_batch.size(0)

        avg_train_loss = train_loss / train_total
        train_acc      = train_correct / train_total * 100

        # — Validate —
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs  = model(X_batch)
                loss     = criterion(outputs, y_batch)
                val_loss    += loss.item() * X_batch.size(0)
                preds        = torch.argmax(outputs, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += X_batch.size(0)

        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total * 100

        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}]  "
            f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.1f}%  |  "
            f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✅ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n🎯 Training complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {MODEL_SAVE}")


if __name__ == "__main__":
    train()