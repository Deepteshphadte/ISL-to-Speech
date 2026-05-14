import cv2
import numpy as np
import mediapipe as mp
import os

SAVE_DIR = "data/processed/NONE"
SEQUENCE_LENGTH = 30
NUM_SAMPLES = 200

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

sequence = []
sample_count = 0

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    print("Creating NONE samples...")
    print("Move hand randomly or keep no hand.")
    print("Press Q to quit.")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        landmarks = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        # Pad to 126
        while len(landmarks) < 126:
            landmarks.append(0)

        landmarks = landmarks[:126]

        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:

            save_path = os.path.join(
                SAVE_DIR,
                f"none_{sample_count}.npy"
            )

            np.save(
                save_path,
                np.array(sequence)
            )

            sample_count += 1

            print(f"Saved NONE sample {sample_count}")

            sequence = []

        cv2.putText(
            frame,
            f"NONE Samples: {sample_count}/{NUM_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("NONE Sample Collection", frame)

        if sample_count >= NUM_SAMPLES:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("DONE creating NONE samples.")