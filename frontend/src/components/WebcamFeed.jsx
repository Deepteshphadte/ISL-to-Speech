import Webcam from "react-webcam";
import { useRef, useEffect, useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

const HAND_CONNECTIONS = [

  [0,1],[1,2],[2,3],[3,4],

  [0,5],[5,6],[6,7],[7,8],

  [5,9],[9,10],[10,11],[11,12],

  [9,13],[13,14],[14,15],[15,16],

  [13,17],[17,18],[18,19],[19,20],

  [0,17]

];

function WebcamFeed({ sendFrame }) {

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const { landmarks } = useContext(PredictionContext);

  // Draw landmarks whenever they change
  useEffect(() => {

    const canvas = canvasRef.current;

    if (!canvas) return;
    const video = webcamRef.current?.video;

    if (!video) return;

    // Match the canvas size to the displayed video
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    const ctx = canvas.getContext("2d");

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    if (!landmarks || landmarks.length === 0) return;

      landmarks.forEach((hand) => {

      // Draw skeleton lines
      HAND_CONNECTIONS.forEach(([start, end]) => {

          const p1 = hand[start];
          const p2 = hand[end];

          ctx.beginPath();

          ctx.moveTo(
              p1.x * canvas.width,
              p1.y * canvas.height
          );

          ctx.lineTo(
              p2.x * canvas.width,
              p2.y * canvas.height
          );

          // Same appearance as MediaPipe/OpenCV
          ctx.strokeStyle = "rgb(224,224,224)";
          ctx.lineWidth = 2;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";

          ctx.stroke();

      });

      // Draw joints
      // Draw joints
      hand.forEach((point) => {

        ctx.beginPath();

        ctx.arc(
            point.x * canvas.width,
            point.y * canvas.height,
            3,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = "rgb(255,0,0)";
        ctx.fill();

    });
  });
  }, [landmarks]);

  // Send webcam frames continuously
  useEffect(() => {

    let animationId;

    const captureFrame = () => {

      if (webcamRef.current) {

        const imageSrc = webcamRef.current.getScreenshot();

        if (imageSrc) {

          sendFrame(imageSrc);

        }

      }

      animationId = requestAnimationFrame(captureFrame);

    };

    captureFrame();

    return () => cancelAnimationFrame(animationId);

  }, [sendFrame]);

  return (

    <div className="camera-box">

      <Webcam
          ref={webcamRef}
          audio={false}
          mirrored={true}
          screenshotFormat="image/jpeg"
          screenshotQuality={0.7}
          forceScreenshotSourceSize={true}
          videoConstraints={{
              width: 640,
              height: 480,
              facingMode: "user"
          }}
          className="webcam"
      />
      <canvas
        ref={canvasRef}
        className="overlay"
      />

    </div>

  );

}

export default WebcamFeed;