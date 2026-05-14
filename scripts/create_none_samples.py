import os
import numpy as np

SAVE_PATH = "data/processed/NONE"

os.makedirs(SAVE_PATH, exist_ok=True)

NUM_SAMPLES = 100
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126

for i in range(NUM_SAMPLES):

    sample = np.zeros(
        (SEQUENCE_LENGTH, FEATURE_SIZE),
        dtype=np.float32
    )

    np.save(
        os.path.join(SAVE_PATH, f"sample_{i}.npy"),
        sample
    )

print(f"Created {NUM_SAMPLES} NONE samples.")