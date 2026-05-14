import cv2
import torch
import time
import pyttsx3
import threading
import numpy as np
import mediapipe as mp
import os
from utils.context_refiner import refine_text
from collections import deque
from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/lstm_model.pth"
LABELS_PATH = "data/labels.npy"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.92
PRED_QUEUE_SIZE = 10
LETTER_COOLDOWN = 20
NO_HAND_RESET = 20
# ──────────────────────────────────────────────────────────────────────────────

# ─── UI Colors ───────────────────────────────────────
BG_COLOR = (20, 20, 20)
PANEL_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 120, 0)
YELLOW = (0, 255, 255)
# ─────────────────────────────────────────────────────

def load_labels():

    if os.path.exists(LABELS_PATH):

        labels = [
            str(l)
            for l in np.load(LABELS_PATH, allow_pickle=True)
        ]

        print(f"Labels loaded from file: {labels}")

    else:

        labels = sorted(os.listdir("data/processed"))

        if "NONE" in labels:
            labels.remove("NONE")
            labels.append("NONE")

        print(f"Labels loaded from folder: {labels}")

    return labels


def load_model(num_classes):

    model = LSTMModel(
        input_size=126,
        hidden_size=256,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.4
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model


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

    if not left_hand:
        left_hand = [0] * 63

    if not right_hand:
        right_hand = [0] * 63

    combined = left_hand + right_hand

    return normalize_landmarks(combined)


def draw_ui(frame, current_sign, sentence, confidence):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    # Top Header
    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 90),
        PANEL_COLOR,
        -1
    )

    # Bottom Panel
    cv2.rectangle(
        overlay,
        (0, h - 140),
        (w, h),
        PANEL_COLOR,
        -1
    )

    # Blend panels
    cv2.addWeighted(
        overlay,
        0.7,
        frame,
        0.3,
        0,
        frame
    )

    # Title
    cv2.putText(
        frame,
        "Indian Sign Language Recognition",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        YELLOW,
        2
    )

    # Current Sign
    color = GREEN if current_sign != "No Sign" else RED

    cv2.putText(
        frame,
        f"Sign: {current_sign}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    # Confidence Bar at Bottom
    if current_sign != "No Sign":

        # Confidence label
        cv2.putText(
            frame,
            "Confidence",
            (20, h - 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            1
        )

        # Background bar
        cv2.rectangle(
            frame,
            (150, h - 140),
            (420, h - 110),
            (70, 70, 70),
            -1
        )

        # Filled confidence bar
        bar_width = int(confidence * 270)

        cv2.rectangle(
            frame,
            (150, h - 140),
            (150 + bar_width, h - 110),
            GREEN,
            -1
        )

        # Percentage text
        cv2.putText(
            frame,
            f"{confidence*100:.1f}%",
            (430, h - 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2
        )
    # Sentence Box
    cv2.putText(
        frame,
        "Recognized Text:",
        (20, h - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        YELLOW,
        2
    )

    cv2.putText(
        frame,
        sentence,
        (20, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        TEXT_COLOR,
        2
    )

    # Controls
    cv2.putText(
        frame,
        "[C] Clear   [Q] Quit",
        (w - 260, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    return frame

def speak_text(text):

    engine = pyttsx3.init()

    engine.setProperty('rate', 150)

    engine.setProperty('volume', 1.0)

    speech_text = text.replace("_", " ")

    engine.say(speech_text)

    engine.runAndWait()

    engine.stop()

def run():

    labels = load_labels()

    num_classes = len(labels)

    model = load_model(num_classes)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print(f"Model loaded. Running on: {device}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    sequence = []
    prediction_queue = deque(maxlen=PRED_QUEUE_SIZE)

    sentence = ""
    last_sign = ""

    no_hand_counter = 0

    current_sign = "No Sign"
    confidence = 0.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    prev_time = 0

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

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # ─────────────────────────────────────────────
            # NO HAND
            # ─────────────────────────────────────────────
            if not results.multi_hand_landmarks:

                current_sign = "No Sign"

                no_hand_counter += 1

                last_sign = ""

                prediction_queue.clear()

                if no_hand_counter >= NO_HAND_RESET:
                    sequence = []

            # ─────────────────────────────────────────────
            # HAND DETECTED
            # ─────────────────────────────────────────────
            else:

                no_hand_counter = 0

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                landmarks = extract_landmarks(results)

                sequence.append(landmarks)

                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                # ─────────────────────────────────────────
                # PREDICTION
                # ─────────────────────────────────────────
                if len(sequence) == SEQUENCE_LENGTH:

                    input_tensor = torch.tensor(
                        np.array(sequence),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():

                        output = model(input_tensor)

                        probs = torch.softmax(output, dim=1)

                        conf, pred_class = torch.max(probs, dim=1)

                    confidence = conf.item()

                    pred_label = labels[pred_class.item()]

                    print(
                        f"Prediction: {pred_label} | "
                        f"Confidence: {confidence:.2f}"
                    )

                    # Ignore weak predictions
                    if confidence < CONFIDENCE_THRESHOLD:

                        current_sign = "No Sign"

                        last_sign = ""

                        prediction_queue.clear()

                    else:

                        prediction_queue.append(pred_label)

                        # Stable prediction
                        final_sign = max(
                            set(prediction_queue),
                            key=prediction_queue.count
                        )

                        if final_sign == "NONE":

                            current_sign = "No Sign"

                            last_sign = ""

                        else:

                            current_sign = final_sign

                            # Add only once
                            if current_sign != last_sign:

                                sentence = refine_text(current_sign)

                                last_sign = current_sign
                                

                                print(
                                    f"Added: {current_sign} | "
                                    f"Sentence: {sentence}"
                                )

                                threading.Thread(
                                    target=speak_text,
                                    args=(sentence,),
                                    daemon=True
                                ).start()

                                # Prevent further additions
                                current_sign = "Waiting..."

            # ─────────────────────────────────────────────
            # DRAW UI
            # ─────────────────────────────────────────────
            frame = draw_ui(
                frame,
                current_sign,
                sentence,
                confidence
            )

            h, w = frame.shape[:2]

            curr_time = time.time()

            fps = 1 / (curr_time - prev_time + 1e-5)

            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (w - 160, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2
            )

            cv2.imshow(
                "ISL Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):

                sentence = ""

                last_sign = ""

                print("Sentence cleared.")

        cap.release()

        cv2.destroyAllWindows()

        print(f"\nFinal sentence: {sentence.strip()}")


if __name__ == "__main__":
    run()