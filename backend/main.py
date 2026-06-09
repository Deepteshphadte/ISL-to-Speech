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
        "status": "online",
        "backend": "FastAPI",
        "model": "not loaded"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    while True:

        image_data = await websocket.receive_text()

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

        result = predictor.predict(
            frame
        )

        await websocket.send_json(
            result
        )