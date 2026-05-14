import cv2
import torch
import numpy as np
import mediapipe as mp
import os
from collections import deque
from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH           = "models/lstm_model.pth"
LABELS_PATH          = "data/labels.npy"
SEQUENCE_LENGTH      = 30
CONFIDENCE_THRESHOLD = 0.40
PRED_QUEUE_SIZE      = 5
LETTER_COOLDOWN      = 10
NO_HAND_RESET        = 30
# ──────────────────────────────────────────────────────────────────────────────


def load_labels():
    if os.path.exists(LABELS_PATH):
        labels = [str(l) for l in np.load(LABELS_PATH, allow_pickle=True)]
        print(f"Labels loaded from file: {labels}")
    else:
        DATA_PATH = "data/processed"
        labels = sorted(os.listdir(DATA_PATH))
        if "NONE" in labels:
            labels.remove("NONE")
            labels.append("NONE")
        print(f"Labels loaded from folder: {labels}")
    return labels


def load_model(num_classes):
    model = LSTMModel(input_size=126, hidden_size=256, num_classes=num_classes, num_layers=2, dropout=0.4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def extract_landmarks(results):
    left_hand, right_hand = [], []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            hand_label = handedness.classification[0].label
            landmarks  = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    if not left_hand:
        left_hand  = [0] * 63
    if not right_hand:
        right_hand = [0] * 63

    combined = left_hand + right_hand
    return normalize_landmarks(combined)


def draw_ui(frame, current_sign, sentence, confidence):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color = (0, 255, 0) if current_sign not in ["No Sign", "NONE"] else (0, 0, 255)
    cv2.putText(frame, f"Sign: {current_sign}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

    if current_sign not in ["No Sign", "NONE"]:
        bar_w = int(confidence * 300)
        cv2.rectangle(frame, (20, 80), (320, 100), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 100), (0, 255, 0), -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (330, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"Sentence: {sentence}", (20, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "[SPACE] Space  [C] Clear  [Q] Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame


def run():
    labels      = load_labels()
    num_classes = len(labels)

    model  = load_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded. Running on: {device}")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    sequence         = []
    prediction_queue = deque(maxlen=PRED_QUEUE_SIZE)
    sentence         = ""
    last_letter      = ""
    cooldown_counter = 0
    no_hand_counter  = 0
    current_sign     = "No Sign"
    confidence       = 0.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        print("Starting prediction... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                no_hand_counter += 1
                if no_hand_counter >= NO_HAND_RESET:
                    sequence = []
                    prediction_queue.clear()
                    current_sign = "No Sign"
                    confidence   = 0.0
            else:
                no_hand_counter = 0

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = extract_landmarks(results)
                sequence.append(landmarks)

                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                if len(sequence) == SEQUENCE_LENGTH:
                    input_tensor = torch.tensor(
                        np.array(sequence), dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs  = torch.softmax(output, dim=1)
                        conf, pred_class = torch.max(probs, dim=1)
                        print(
                            f"Prediction: {labels[pred_class.item()]} | "
                            f"Confidence: {conf.item():.2f}"
                        )

                    confidence = conf.item()
                    pred_class = pred_class.item()

                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_queue.append(pred_class)

            if len(prediction_queue) > 0:
                final_class  = max(set(prediction_queue), key=prediction_queue.count)
                current_sign = labels[final_class]
                if current_sign == "NONE":
                    current_sign = "No Sign"
            else:
                current_sign = "No Sign"

            cooldown_counter = max(0, cooldown_counter - 1)

            if (
                current_sign != "No Sign"
                and current_sign != last_letter
                and cooldown_counter == 0
            ):
                sentence        += current_sign + " "
                last_letter      = current_sign
                cooldown_counter = LETTER_COOLDOWN
                print(f"Added: {current_sign}  |  Sentence: {sentence}")

            frame = draw_ui(frame, current_sign, sentence.strip(), confidence)
            cv2.imshow("ISL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                sentence    += " "
                last_letter  = ""
                print(f"Space added  |  Sentence: {sentence}")
            elif key == ord('c'):
                sentence     = ""
                last_letter  = ""
                print("Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal sentence: {sentence.strip()}")


if __name__ == "__main__":
    run()