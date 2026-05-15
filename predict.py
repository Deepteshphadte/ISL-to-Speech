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
PANEL_COLOR = (10, 10, 10)
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


def draw_ui(frame, current_sign, sentence, confidence, fps):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    # ─── TOP BAR ─────────────────────────────
    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 70),
        (10, 10, 10),
        -1
    )

    # ─── BOTTOM BAR ──────────────────────────
    cv2.rectangle(
        overlay,
        (0, h - 110),
        (w, h),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0,
        frame
    )

    # ─── TITLE ───────────────────────────────
    cv2.putText(
        frame,
        "ISL to Speech Converter",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        YELLOW,
        2
    )

    # ─── STATUS ──────────────────────────────
    status = "ACTIVE" if current_sign != "No Sign" else "IDLE"

    status_color = GREEN if status == "ACTIVE" else RED

    cv2.putText(
        frame,
        f"Status: {status}",
        (w - 220, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        status_color,
        1
    )

    # ─── FPS ─────────────────────────────────
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w - 130, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        GREEN,
        1
    )

    # ─── CURRENT SIGN BOX ────────────────────
    cv2.rectangle(
        frame,
        (20, 85),
        (180, 165),
        (40, 40, 40),
        -1
    )

    cv2.rectangle(
        frame,
        (20, 85),
        (180, 165),
        (180, 180, 180),
        1
    )

    cv2.putText(
        frame,
        "Current Sign",
        (32, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        YELLOW,
        1
    )

    sign_color = GREEN if current_sign != "No Sign" else RED

    cv2.putText(
        frame,
        current_sign,
        (32, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        sign_color,
        2
    )

    # ─── CONFIDENCE ──────────────────────────
    if current_sign != "No Sign":

        cv2.putText(
            frame,
            "Confidence",
            (20, h - 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1
        )

        # Bar outline
        cv2.rectangle(
            frame,
            (120, h - 82),
            (300, h - 62),
            (255, 255, 255),
            1
        )

        # Filled confidence
        bar_width = int(confidence * 180)

        bar_color = GREEN if confidence > 0.9 else RED

        cv2.rectangle(
            frame,
            (120, h - 82),
            (120 + bar_width, h - 62),
            bar_color,
            -1
        )

        cv2.putText(
            frame,
            f"{confidence*100:.0f}%",
            (315, h - 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1
        )

    # ─── RECOGNIZED TEXT ─────────────────────
    cv2.putText(
        frame,
        "Recognized:",
        (20, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        YELLOW,
        1
    )

    cv2.putText(
        frame,
        sentence[-35:],
        (140, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1
    )

    # ─── CONTROLS ────────────────────────────
    cv2.putText(
        frame,
        "[C] Clear",
        (w - 180, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1
    )

    cv2.putText(
        frame,
        "[Q] Quit",
        (w - 180, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
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
    sentence_buffer = []
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

        cv2.namedWindow(
            "ISL Recognition",
            cv2.WND_PROP_FULLSCREEN
        )

        cv2.setWindowProperty(
            "ISL Recognition",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

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

                confidence = 0.0

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

                    raw_confidence = conf.item()

                    # Live dynamic confidence for UI
                    confidence = raw_confidence

                    pred_label = labels[pred_class.item()]

                    print(
                        f"Prediction: {pred_label} | "
                        f"Confidence: {confidence:.2f}"
                    )

                    # Ignore weak predictions
                    if raw_confidence < CONFIDENCE_THRESHOLD:

                        current_sign = "No Sign"

                        confidence = 0.0

                        last_sign = ""

                        prediction_queue.clear()

                    else:

                        prediction_queue.append(pred_label)
                        confidence = raw_confidence

                        # Stable prediction
                        final_sign = max(
                            set(prediction_queue),
                            key=prediction_queue.count
                        )

                        if final_sign == "NONE":

                            current_sign = "No Sign"

                            confidence = 0.0

                            last_sign = ""

                        else:

                            current_sign = final_sign

                            # Add only once
                            if current_sign != last_sign:

                                sentence = refine_text(current_sign)

                                last_sign = current_sign
                                if len(sentence_buffer) == 0 or sentence_buffer[-1] != current_sign:
                                     sentence_buffer.append(current_sign)
                                sentence = refine_text(sentence_buffer)

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
            curr_time = time.time()

            fps = 1 / (curr_time - prev_time + 1e-5)

            prev_time = curr_time

            
            frame = draw_ui(
                frame,
                current_sign,
                sentence,
                confidence,
                fps
            )

            h, w = frame.shape[:2]

            
            status = "ACTIVE" if current_sign != "No Sign" else "IDLE"

            status_color = GREEN if "ACTIVE" in status else RED


            cv2.imshow(
                "ISL Recognition",
                frame
            )
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):

                sentence = ""

                sentence_buffer.clear()

                last_sign = ""

                print("Sentence cleared.")

        cap.release()

        cv2.destroyAllWindows()

        print(f"\nFinal sentence: {sentence.strip()}")


if __name__ == "__main__":
    run()