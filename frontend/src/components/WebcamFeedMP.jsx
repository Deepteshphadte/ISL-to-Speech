import { useEffect, useRef } from "react";
import Webcam from "react-webcam";

import {
  DrawingUtils,
  HandLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

function WebcamFeedMP({ sendFrame }) {

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const handLandmarkerRef = useRef(null);
  const animationRef = useRef();

  useEffect(() => {

    const loadModel = async () => {

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      handLandmarkerRef.current =
        await HandLandmarker.createFromOptions(
          vision,
          {

            baseOptions: {

              modelAssetPath: "/hand_landmarker.task"

            },

            runningMode: "VIDEO",

            numHands: 2

          }
        );

      console.log("✅ MediaPipe Loaded");
      setTimeout(() => {
        detectHands();
      }, 500);

    };

    loadModel();

  }, []);

  const detectHands = () => {

  const video = webcamRef.current?.video;
  const canvas = canvasRef.current;

  if (!video || !canvas || !handLandmarkerRef.current) {
    animationRef.current = requestAnimationFrame(detectHands);
    return;
  }

  if (video.readyState < 2) {
    animationRef.current = requestAnimationFrame(detectHands);
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const results = handLandmarkerRef.current.detectForVideo(
    video,
    performance.now()
  );



  if (results.landmarks) {

    const drawingUtils = new DrawingUtils(ctx);

    for (const landmarks of results.landmarks) {

      drawingUtils.drawConnectors(
        landmarks,
        HandLandmarker.HAND_CONNECTIONS,
        {
          color: "#00FF00",
          lineWidth: 3
        }
      );

      drawingUtils.drawLandmarks(
        landmarks,
        {
          color: "#FF0000",
          radius: 4
        }
      );

    }

  }

  animationRef.current = requestAnimationFrame(detectHands);

};

  useEffect(() => {

    return () => {

      cancelAnimationFrame(
        animationRef.current
      );

    };

  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!webcamRef.current || !sendFrame) return;

      const frame = webcamRef.current.getScreenshot();

      if (frame) {
        sendFrame(frame);
      }
    }, 200); // 5 FPS to backend

    return () => clearInterval(interval);
  }, [sendFrame]);

  return (

    <div className="camera-box">

      <Webcam
        ref={webcamRef}
        audio={false}
        mirrored={true}
        className="webcam"
        screenshotFormat="image/jpeg"
        videoConstraints={{
          width: 640,
          height: 480,
          facingMode: "user"
        }}
      />

      <canvas
        ref={canvasRef}
        className="overlay"
      />

    </div>

  );

}

export default WebcamFeedMP;