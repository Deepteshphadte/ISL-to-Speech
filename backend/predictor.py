import os
import time
import threading
from queue import Queue
from collections import deque

import torch
import numpy as np
import cv2
import mediapipe as mp

from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks
from backend.gemini_service import refine_sentence
from backend.speech_engine import SpeechEngine

# ─── Config — MUST match your training data shape ────────────────────────────
MODEL_PATH = "models/lstm_model.pth"
LABELS_PATH = "data/labels.npy"

SEQUENCE_LENGTH = 10          # training data shape is (N, 30, 126) — keep in sync
CONFIDENCE_THRESHOLD = 0.92   # same as predict.py, which detects correctly
PRED_QUEUE_SIZE = 10
NO_HAND_RESET = 20            # frames of sustained no-hand before the LSTM sequence buffer clears

# ─── Transition stability (fixes repeated speech when switching signs) ──────
# A single dropped frame during the quick motion of switching from one sign
# to the next used to instantly wipe sign state and cause the previous sign
# to be re-announced. These add hysteresis so brief noise doesn't do that.
NO_HAND_GRACE_FRAMES = 5       # tolerate this many no-hand frames before resetting sign state
MIN_VOTE_RATIO = 0.6           # fraction of the prediction queue that must agree before a sign is accepted
# ──────────────────────────────────────────────────────────────────────────────

# ─── Phase 2: sentence-level Gemini finalization ─────────────────────────────
AUTO_FINALIZE_SECONDS = 2.5

# Speak each confirmed sign immediately (e.g. say "A" the moment it's
# confirmed), in addition to speaking the final Gemini-refined sentence
# once the user pauses.
SPEAK_INDIVIDUAL_SIGNS = True
# ──────────────────────────────────────────────────────────────────────────────


class Predictor:

    def __init__(self):
        self.auto_speak = True

        self.labels = self.load_labels()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model(len(self.labels))
        self.model.to(self.device)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.sequence = []
        self.prediction_queue = deque(maxlen=PRED_QUEUE_SIZE)

        self.sentence_buffer = []
        self.refined_sentence = ""

        self.last_sign = ""
        self.current_sign = "No Sign"
        self.confidence = 0.0

        self.no_hand_counter = 0

        # ─── Phase 2 finalization state ───────────────────
        self.last_word_time = 0.0
        self.sentence_finalized = True   # no sentence currently pending
        # ────────────────────────────────────────────────────

        self.speech = SpeechEngine()

        print("✅ Predictor Initialized")

        # ─── Queues / workers ──────────────────────────────
        self.gemini_queue = Queue()
        self.speech_queue = Queue()

        threading.Thread(target=self.gemini_worker, daemon=True).start()
        threading.Thread(target=self.speech_worker, daemon=True).start()
        # ────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # Setup helpers
    # ─────────────────────────────────────────────────────────────────

    def load_labels(self):
        if os.path.exists(LABELS_PATH):
            labels = [str(label) for label in np.load(LABELS_PATH, allow_pickle=True)]
        else:
            labels = sorted(os.listdir("data/processed"))
            if "NONE" in labels:
                labels.remove("NONE")
                labels.append("NONE")

        print(f"✅ Labels Loaded ({len(labels)}): {labels}")
        return labels

    def load_model(self, num_classes):
        model = LSTMModel(
            input_size=126,
            hidden_size=256,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.4
        )

        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()

        print("✅ LSTM Model Loaded")
        return model

    def extract_landmarks(self, results):
        left_hand = []
        right_hand = []

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

    def _build_display_landmarks(self, results):
        display_landmarks = []
        for hand in results.multi_hand_landmarks:
            one_hand = [{"x": lm.x, "y": lm.y} for lm in hand.landmark]
            display_landmarks.append(one_hand)
        return display_landmarks

    def _base_response(self, sign, confidence, landmarks):
        return {
            "sign": sign,
            "confidence": confidence,
            "sentence": " ".join(self.sentence_buffer),
            "refined_sentence": self.refined_sentence,
            "landmarks": landmarks
        }

    def _say(self, text):
        """Queue text to be spoken. Never calls speech.speak() directly —
        everything goes through the single speech_worker so calls never
        overlap."""
        if self.auto_speak and text:
            self.speech_queue.put(text)

    # ─────────────────────────────────────────────────────────────────
    # Background workers
    # ─────────────────────────────────────────────────────────────────

    def gemini_worker(self):
        while True:
            raw_sentence = self.gemini_queue.get()

            try:
                refined = refine_sentence(raw_sentence)
                self.refined_sentence = refined
                print("Gemini:", refined)

                self._say(refined)

            except Exception as e:
                print("Gemini worker error:", e)

            self.gemini_queue.task_done()

    def speech_worker(self):
        while True:
            text = self.speech_queue.get()

            try:
                if text:
                    self.speech.speak(text)
            except Exception as e:
                print("Speech worker error:", e)

            self.speech_queue.task_done()

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: finalize the sentence once the user pauses signing
    # ─────────────────────────────────────────────────────────────────

    def check_auto_finalize(self):
        if self.sentence_finalized:
            return

        if len(self.sentence_buffer) == 0:
            self.sentence_finalized = True
            return

        if time.time() - self.last_word_time >= AUTO_FINALIZE_SECONDS:
            raw_sentence = " ".join(self.sentence_buffer)
            print(f"⏸️ Pause detected — finalizing sentence: {raw_sentence}")

            self.gemini_queue.put(raw_sentence)

            self.sentence_buffer = []
            self.last_sign = ""
            self.sentence_finalized = True

    # ─────────────────────────────────────────────────────────────────
    # Main prediction loop
    # ─────────────────────────────────────────────────────────────────

    def predict(self, frame):
        self.check_auto_finalize()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # ─── No hand detected ───────────────────────────────
        if not results.multi_hand_landmarks:
            self.no_hand_counter += 1

            # Brief dropout (e.g. mid-transition between two signs) — don't
            # wipe sign state yet, just report "no sign" for this frame
            # without resetting last_sign / the prediction queue.
            if self.no_hand_counter <= NO_HAND_GRACE_FRAMES:
                return self._base_response("No Sign", 0.0, [])

            # Sustained no-hand — genuinely reset
            self.current_sign = "No Sign"
            self.confidence = 0.0
            self.last_sign = ""
            self.prediction_queue.clear()

            if self.no_hand_counter >= NO_HAND_RESET:
                self.sequence = []

            return self._base_response("No Sign", 0.0, [])

        # ─── Hand(s) detected ───────────────────────────────
        self.no_hand_counter = 0

        landmarks = self.extract_landmarks(results)
        display_landmarks = self._build_display_landmarks(results)

        self.sequence.append(landmarks)
        if len(self.sequence) > SEQUENCE_LENGTH:
            self.sequence.pop(0)

        if len(self.sequence) < SEQUENCE_LENGTH:
            return self._base_response("Waiting...", 0.0, display_landmarks)

        input_tensor = torch.tensor(
            np.array(self.sequence), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_class = torch.max(probs, dim=1)

        raw_confidence = conf.item()
        pred_label = self.labels[pred_class.item()]

        print("Prediction:", pred_label, "Confidence:", round(raw_confidence, 3))

        if raw_confidence < CONFIDENCE_THRESHOLD:
            return self._base_response("No Sign", round(raw_confidence, 2), display_landmarks)

        self.prediction_queue.append(pred_label)

        final_sign = max(
            set(self.prediction_queue),
            key=self.prediction_queue.count
        )

        # ─── Require a strong majority, not just a bare plurality ───
        # During a transition, the queue can be a mix of the old and new
        # sign (e.g. 4x "A", 6x "B"). Without this check, "B" would win
        # narrowly on almost every frame of the transition, causing
        # jittery/early confirmations. Requiring most of the queue to
        # agree makes the confirmed sign much more stable.
        vote_count = self.prediction_queue.count(final_sign)
        min_required = max(1, int(len(self.prediction_queue) * MIN_VOTE_RATIO))

        if vote_count < min_required:
            return self._base_response("Detecting...", round(raw_confidence, 2), display_landmarks)
        # ──────────────────────────────────────────────────────────────

        if final_sign == "NONE":
            return self._base_response("No Sign", 0.0, display_landmarks)

        self.current_sign = final_sign
        self.confidence = raw_confidence

        if self.current_sign != self.last_sign:
            self.last_sign = self.current_sign

            if len(self.sentence_buffer) == 0 or self.sentence_buffer[-1] != self.current_sign:
                self.sentence_buffer.append(self.current_sign)

                self.last_word_time = time.time()
                self.sentence_finalized = False

                print("Buffered:", " ".join(self.sentence_buffer))

                if SPEAK_INDIVIDUAL_SIGNS:
                    self._say(self.current_sign.replace("_", " "))

        return self._base_response(
            self.current_sign, round(self.confidence, 2), display_landmarks
        )
