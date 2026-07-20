import base64
import cv2
import numpy as np
import backend.model_loader
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
from backend.predictor import Predictor

app = FastAPI()
predictor = Predictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "ISL Backend Running"
    }

@app.get("/status")
def status():
    return {
        "backend": "online",
        "model": "loaded",
        "camera": "active"
    }

@app.post("/clear")
def clear_sentence():

    predictor.sentence_buffer.clear()
    predictor.prediction_queue.clear()
    predictor.sequence.clear()

    predictor.last_sign = ""
    predictor.current_sign = "No Sign"
    predictor.confidence = 0.0

    predictor.refined_sentence = ""
    predictor.last_spoken_sentence = ""

    return {
        "message": "cleared"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    while True:

        image_data = await websocket.receive_text()

        print("Frame Received")

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(
            image_data
        )

        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        print("Frame Received")

        result = predictor.predict(frame)

        print("Frame Sent")

        print("Sending:", result["sign"])

        print("Frame Sent")

        await websocket.send_json(result)
    
@app.post("/toggle_speech")
def toggle_speech():

    predictor.auto_speak = not predictor.auto_speak

    return {
        "auto_speak": predictor.auto_speak
    }