import cv2
import os
import torch
import numpy as np
import mediapipe as mp

from collections import deque

from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks

MODEL_PATH = "models/lstm_model.pth"
LABELS_PATH = "data/labels.npy"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.92
PRED_QUEUE_SIZE = 10
NO_HAND_RESET = 20


class Predictor:

    def __init__(self):

        self.labels = self.load_labels()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = self.load_model(
            len(self.labels)
        )

        self.model.to(self.device)

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )


        self.sequence = []

        self.prediction_queue = deque(
            maxlen=PRED_QUEUE_SIZE
        )

        self.sentence_buffer = []

        self.last_sign = ""

        self.current_sign = "No Sign"

        self.confidence = 0.0

        self.no_hand_counter = 0

        print("✅ Predictor Initialized")

    def load_labels(self):

        if os.path.exists(LABELS_PATH):

            labels = [
                str(label)
                for label in np.load(
                    LABELS_PATH,
                    allow_pickle=True
                )
            ]

        else:

            labels = sorted(
                os.listdir(
                    "data/processed"
                )
            )

        print(
            f"✅ Labels Loaded ({len(labels)})"
        )

        return labels

    def load_model(self, num_classes):

        model = LSTMModel(
            input_size=126,
            hidden_size=256,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.4
        )

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        model.eval()

        print("✅ LSTM Model Loaded")

        return model

    def extract_landmarks(self, results):

        left_hand = []
        right_hand = []

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                hand_label = (
                    handedness
                    .classification[0]
                    .label
                )

                landmarks = []

                for lm in hand_landmarks.landmark:

                    landmarks.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                if hand_label == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks

        if not left_hand:
            left_hand = [0] * 63

        if not right_hand:
            right_hand = [0] * 63

        combined = (
            left_hand +
            right_hand
        )

        return normalize_landmarks(
            combined
        )

    def predict(self, frame):


       # frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)
        landmarks_to_send = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                hand_points = []

                for lm in hand.landmark:

                    hand_points.append({
                        "x": lm.x,
                        "y": lm.y
                    })

                landmarks_to_send.append(hand_points)
            print("HAND DETECTED")
        else:
            print("NO HAND")

        if not results.multi_hand_landmarks:

            self.current_sign = "No Sign"

            self.confidence = 0.0

            self.last_sign = ""

            self.prediction_queue.clear()

            self.no_hand_counter += 1

            if (
                self.no_hand_counter
                >= NO_HAND_RESET
            ):
                self.sequence = []

            return {
                "sign": self.current_sign,
                "confidence": self.confidence,
                "sentence": " ".join(
                    self.sentence_buffer
                ),
                "landmarks": landmarks_to_send
            }

        self.no_hand_counter = 0

        landmarks = self.extract_landmarks(
            results
        )
        display_landmarks = []

        for hand in results.multi_hand_landmarks:

            one_hand = []

            for lm in hand.landmark:

                one_hand.append({
                    "x": lm.x,
                    "y": lm.y
                })

            display_landmarks.append(one_hand)

        self.sequence.append(
            landmarks
        )
        print("Sequence Length:", len(self.sequence))

        if len(self.sequence) > SEQUENCE_LENGTH:

            self.sequence.pop(0)

        if len(self.sequence) < SEQUENCE_LENGTH:

            return {
                "sign": "Waiting...",
                "confidence": 0.0,
                "sentence": " ".join(
                    self.sentence_buffer
                ),
                "landmarks": display_landmarks
            }

        input_tensor = torch.tensor(
            np.array(self.sequence),
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

            probs = torch.softmax(
                output,
                dim=1
            )

            conf, pred_class = torch.max(
                probs,
                dim=1
            )

        raw_confidence = conf.item()

        pred_label = self.labels[
            pred_class.item()
        ]
        print(
            "Prediction:",
            pred_label,
            "Confidence:",
            raw_confidence
        )

        if (
            raw_confidence
            < CONFIDENCE_THRESHOLD
        ):

            return {
                "sign": "No Sign",
                "confidence": 0.0,
                "sentence": " ".join(
                    self.sentence_buffer
                ),
                "landmarks": []
            }

        self.prediction_queue.append(
            pred_label
        )

        final_sign = max(
            set(self.prediction_queue),
            key=self.prediction_queue.count
        )

        if final_sign == "NONE":

            return {
                "sign": "No Sign",
                "confidence": 0.0,
                "sentence": " ".join(
                    self.sentence_buffer
                ),
                "landmarks": []
            }

        self.current_sign = final_sign

        self.confidence = raw_confidence

        if (
            self.current_sign
            != self.last_sign
        ):

            self.last_sign = (
                self.current_sign
            )

            if (
                len(self.sentence_buffer)
                == 0
                or
                self.sentence_buffer[-1]
                != self.current_sign
            ):

                self.sentence_buffer.append(
                    self.current_sign
                )

        return {
            "sign": self.current_sign,
            "confidence": round(
                self.confidence,
                2
            ),
            "sentence": " ".join(
                self.sentence_buffer
            ),
            "landmarks": display_landmarks
        }