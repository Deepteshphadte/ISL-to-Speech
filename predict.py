import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from models.lstm_model import LSTMModel
from utils.preprocessing import normalize_landmarks

# Load model
model = LSTMModel(num_classes=4)
model.load_state_dict(torch.load("models/lstm_model.pth"))
model.eval()

labels = ["A", "B", "C", "No Sign"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30
sequence = []

# 🔥 Smooth predictions
prediction_queue = deque(maxlen=10)

def extract_landmarks(results):
    left_hand, right_hand = [], []

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
        left_hand = [0]*63
    if not right_hand:
        right_hand = [0]*63

    combined = left_hand + right_hand
    return normalize_landmarks(combined)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # 🔥 Reset sequence if no hand
        if not results.multi_hand_landmarks:
            sequence = []
            prediction_queue.clear()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(results)
            sequence.append(landmarks)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
                prediction = model(input_data)

                probs = torch.softmax(prediction, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

                confidence = confidence.item()
                predicted_class = predicted_class.item()

                # 🔥 Confidence threshold
                if confidence > 0.75:
                    prediction_queue.append(predicted_class)

        # 🔥 Majority voting (stability)
        if len(prediction_queue) > 0:
            final_class = max(set(prediction_queue), key=prediction_queue.count)
            text = labels[final_class]
        else:
            text = "No Sign"

        # Display
        cv2.putText(frame, text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0,255,0) if text != "No Sign" else (0,0,255), 3)

        cv2.imshow("Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()