import os
import cv2
import numpy as np
import mediapipe as mp
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import normalize_landmarks

# ─── CONFIG ─────────────────────────────────────────────────────
DATASET_PATH = "dataset/words"
OUTPUT_PATH = "data/processed"

SEQUENCE_LENGTH = 30
# ────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands


def extract_landmarks(results):
    left_hand = []
    right_hand = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_label = handedness.classification[0].label
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    # Fill missing hands with zeros
    if not left_hand:
        left_hand = [0] * 63

    if not right_hand:
        right_hand = [0] * 63

    combined = left_hand + right_hand

    return normalize_landmarks(combined)


def process_video(video_path, hands):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        landmarks = extract_landmarks(results)

        frames.append(landmarks)

    cap.release()

    # If no frames found
    if len(frames) == 0:
        return None

    frames = np.array(frames)

    # ─── Make Exactly 30 Frames ─────────────────────────

    if len(frames) > SEQUENCE_LENGTH:
        # Sample evenly
        indices = np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        frames = frames[indices]

    elif len(frames) < SEQUENCE_LENGTH:
        # Pad with last frame
        padding = np.repeat(
            frames[-1][np.newaxis, :],
            SEQUENCE_LENGTH - len(frames),
            axis=0
        )

        frames = np.vstack((frames, padding))

    return frames.astype(np.float32)


def main():

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        # Loop through gesture folders
        for label in os.listdir(DATASET_PATH):

            label_path = os.path.join(DATASET_PATH, label)

            if not os.path.isdir(label_path):
                continue

            print(f"\nProcessing label: {label}")

            save_path = os.path.join(OUTPUT_PATH, label)
            os.makedirs(save_path, exist_ok=True)

            sample_count = 0

            for file in os.listdir(label_path):

                if not file.lower().endswith((".mp4", ".mov", ".avi")):
                    continue

                video_path = os.path.join(label_path, file)

                print(f"  Processing: {file}")

                sequence = process_video(video_path, hands)

                if sequence is None:
                    print("    Skipped (no frames)")
                    continue

                save_file = os.path.join(
                    save_path,
                    f"sample_{sample_count}.npy"
                )

                np.save(save_file, sequence)

                print(f"    Saved: sample_{sample_count}.npy")

                sample_count += 1

            print(f"Completed {label}: {sample_count} samples")


if __name__ == "__main__":
    main()