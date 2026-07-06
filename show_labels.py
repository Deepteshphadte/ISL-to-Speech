import numpy as np

labels = np.load(
    "data/labels.npy",
    allow_pickle=True
)

print("\n===== LABELS =====\n")

for i, label in enumerate(labels):
    print(f"{i} : {label}")