import Webcam from "react-webcam";
import { useRef, useEffect } from "react";

function WebcamFeed({ sendFrame }) {

  const webcamRef = useRef(null);

  useEffect(() => {

    const interval = setInterval(() => {

      if (webcamRef.current) {

        const imageSrc =
          webcamRef.current.getScreenshot();

        if (imageSrc) {

          sendFrame(imageSrc);

        }
      }

    }, 100);

    return () => clearInterval(interval);

  }, []);

  return (
    <div className="card">
      <h2>Live Camera</h2>

      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        mirrored={true}
        className="webcam"
    />
    </div>
  );
}

export default WebcamFeed;