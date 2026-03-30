import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalize landmarks:
    - Center around wrist
    - Scale by max distance
    """

    landmarks = np.array(landmarks)

    # reshape (126 → 42 x 3)
    landmarks = landmarks.reshape(-1, 3)

    # use first point (wrist) as origin
    origin = landmarks[0]

    # shift
    landmarks = landmarks - origin

    # scale
    max_val = np.max(np.abs(landmarks))
    if max_val != 0:
        landmarks = landmarks / max_val

    return landmarks.flatten()