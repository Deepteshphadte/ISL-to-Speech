import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_PATH = "data/processed"
SEQUENCE_LENGTH = 30  # frames per sample
COUNTDOWN_SECONDS = 3  # seconds before recording starts


def extract_landmarks(results):
    left_hand = []
    right_hand = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            hand_label = handedness.classification[0].label
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    if not left_hand:
        left_hand = [0] * 63
    if not right_hand:
        right_hand = [0] * 63

    from utils.preprocessing import normalize_landmarks

    combined = left_hand + right_hand
    return normalize_landmarks(combined)


def collect_data(label="A"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    sequence = []

    save_path = os.path.join(DATA_PATH, label)
    os.makedirs(save_path, exist_ok=True)

    existing_files = [f for f in os.listdir(save_path) if f.endswith(".npy")]
    sample_count = len(existing_files)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        print(f"Collecting data for: {label}")
        print("Press 's' to start countdown, 'q' to quit")

        recording = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1) & 0xFF

            # ── Countdown before recording ──
            if key == ord('s'):
                for countdown in range(COUNTDOWN_SECONDS, 0, -1):
                    ret, frame_cd = cap.read()
                    if not ret:
                        break
                    frame_cd = cv2.flip(frame_cd, 1)

                    # Draw hand landmarks during countdown too
                    rgb_cd = cv2.cvtColor(frame_cd, cv2.COLOR_BGR2RGB)
                    results_cd = hands.process(rgb_cd)
                    if results_cd.multi_hand_landmarks:
                        for hand_landmarks in results_cd.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame_cd, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Big countdown number
                    cv2.putText(frame_cd, str(countdown), (230, 280),
                                cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 20)

                    # Instructions
                    cv2.putText(frame_cd, "Get Ready!", (160, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                    cv2.putText(frame_cd, f"Label: {label}  |  Sample: {sample_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    cv2.imshow("Data Collection", frame_cd)
                    cv2.waitKey(1000)  # wait 1 second per countdown step

                recording = True
                sequence = []
                print("Recording started...")

            # ── Recording frames ──
            if recording:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = extract_landmarks(results)
                sequence.append(landmarks)

                # Show recording indicator
                progress = int((len(sequence) / SEQUENCE_LENGTH) * 300)
                cv2.rectangle(frame, (10, 50), (310, 80), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 50), (10 + progress, 80), (0, 255, 0), -1)
                cv2.putText(frame, f"Recording... {len(sequence)}/{SEQUENCE_LENGTH}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

                if len(sequence) == SEQUENCE_LENGTH:
                    file_name = f"sample_{sample_count}.npy"
                    np.save(os.path.join(save_path, file_name), sequence)
                    print(f"Saved {file_name}  |  Total: {sample_count + 1}")
                    sample_count += 1
                    recording = False
                    sequence = []

                continue  # skip the display below while recording

            if key == ord('q'):
                break

            # ── Normal display (not recording) ──
            cv2.putText(frame, f"Label: {label}  |  Samples: {sample_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to record  |  'q' to quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()