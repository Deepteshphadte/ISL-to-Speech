import os
import numpy as np

DATA_PATH = "data/processed"

# Get labels (A, B, C)
labels = sorted(os.listdir(DATA_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}

X = []
y = []

for label in labels:
    folder_path = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file))
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Labels:", label_map)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Save dataset
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Dataset saved successfully ✅")