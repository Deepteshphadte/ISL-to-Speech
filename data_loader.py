import os
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = "data/processed"
SAVE_PATH = "data"
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset():
    # Get all label folders sorted
    all_labels = sorted(os.listdir(DATA_PATH))

    # Move NONE to end if it exists (so index is always last)
    if "NONE" in all_labels:
        all_labels.remove("NONE")
        all_labels.append("NONE")

    label_map = {label: idx for idx, label in enumerate(all_labels)}

    print(f"Labels found : {all_labels}")
    print(f"Label map    : {label_map}")

    X, y = [], []
    skipped = 0

    for label in all_labels:
        folder_path = os.path.join(DATA_PATH, label)

        npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

        if len(npy_files) == 0:
            print(f"  [WARNING] No samples found for label: {label}")
            continue

        for file in npy_files:
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)

            # Validate shape — must be (30, 126)
            if data.shape != (30, 126):
                print(f"  [SKIP] Bad shape {data.shape} in {file_path}")
                skipped += 1
                continue

            X.append(data)
            y.append(label_map[label])

        print(f"  [{label}] Loaded {len(npy_files)} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nTotal samples : {len(X)}")
    print(f"Skipped       : {skipped}")
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")

    # Save
    np.save(os.path.join(SAVE_PATH, "X.npy"), X)
    np.save(os.path.join(SAVE_PATH, "y.npy"), y)

    # Save label map for use in predict.py
    label_list = all_labels  # ordered list
    np.save(os.path.join(SAVE_PATH, "labels.npy"), label_list)

    print(f"\n✅ Dataset saved to {SAVE_PATH}/")
    print(f"   X.npy | y.npy | labels.npy")

    return X, y, label_map


if __name__ == "__main__":
    load_dataset()