import cv2
import torch
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import os

from collections import deque
from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks
from utils.context_refiner import refine_text


MODEL_PATH = "models/lstm_model.pth"
LABELS_PATH = "data/labels.npy"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75
PRED_QUEUE_SIZE = 10
NO_HAND_RESET = 30


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)


def speak_text(text):
    speech_text = text.replace("_", " ")
    engine.say(speech_text)
    engine.runAndWait()


# Load labels
labels = [str(l) for l in np.load(LABELS_PATH, allow_pickle=True)]


# Load model
model = LSTMModel(
    input_size=126,
    hidden_size=256,
    num_classes=len(labels),
    num_layers=2,
    dropout=0.4
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils



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



def generate_frames():

    cap = cv2.VideoCapture(0)

    sequence = []
    prediction_queue = deque(maxlen=PRED_QUEUE_SIZE)

    sentence = ""
    sentence_buffer = []

    current_sign = "No Sign"
    confidence = 0.0

    last_sign = ""
    no_hand_counter = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            h, w = frame.shape[:2]

            # UI Background
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 10), -1)
            cv2.rectangle(overlay, (0, h - 110), (w, h), (10, 10, 10), -1)

            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Title
            cv2.putText(
                frame,
                "ISL to Speech Converter",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            # No Hand
            if not results.multi_hand_landmarks:

                current_sign = "No Sign"
                confidence = 0.0

                no_hand_counter += 1

                if no_hand_counter >= NO_HAND_RESET:
                    sequence = []
                    prediction_queue.clear()

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

                if len(sequence) == SEQUENCE_LENGTH:

                    input_tensor = torch.tensor(
                        np.array(sequence),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    with torch.no_grad():

                        output = model(input_tensor)

                        probs = torch.softmax(output, dim=1)

                        conf, pred_class = torch.max(probs, dim=1)

                    raw_confidence = conf.item()

                    confidence = raw_confidence

                    pred_label = labels[pred_class.item()]

                    if raw_confidence > CONFIDENCE_THRESHOLD:

                        prediction_queue.append(pred_label)

                        final_sign = max(
                            set(prediction_queue),
                            key=prediction_queue.count
                        )

                        if final_sign == "NONE":

                            current_sign = "No Sign"
                            confidence = 0.0

                        else:

                            current_sign = final_sign

                            if current_sign != last_sign:

                                last_sign = current_sign

                                if (
                                    len(sentence_buffer) == 0
                                    or sentence_buffer[-1] != current_sign
                                ):
                                    sentence_buffer.append(current_sign)

                                sentence = refine_text(sentence_buffer)

                                threading.Thread(
                                    target=speak_text,
                                    args=(sentence,),
                                    daemon=True
                                ).start()

                    else:

                        current_sign = "No Sign"
                        confidence = 0.0

            # Status
            status = "ACTIVE" if current_sign != "No Sign" else "IDLE"

            status_color = (
                (0, 255, 0)
                if status == "ACTIVE"
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (w - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                status_color,
                1
            )

            # Current Sign Box
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
                (0, 255, 255),
                1
            )

            sign_color = (
                (0, 255, 0)
                if current_sign != "No Sign"
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                current_sign,
                (32, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                sign_color,
                2
            )

            # Confidence
            if current_sign != "No Sign":

                cv2.putText(
                    frame,
                    "Confidence",
                    (20, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )

                cv2.rectangle(
                    frame,
                    (120, h - 82),
                    (300, h - 62),
                    (255, 255, 255),
                    1
                )

                bar_width = int(confidence * 180)

                cv2.rectangle(
                    frame,
                    (120, h - 82),
                    (120 + bar_width, h - 62),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"{confidence*100:.0f}%",
                    (315, h - 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

            # Sentence
            cv2.putText(
                frame,
                "Recognized:",
                (20, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            cv2.putText(
                frame,
                sentence[-35:],
                (140, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            # Encode Frame
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )