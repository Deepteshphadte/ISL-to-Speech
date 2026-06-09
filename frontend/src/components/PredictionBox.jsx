import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function PredictionBox() {

  const {
    currentSign,
    confidence
  } = useContext(PredictionContext);

  return (
    <div className="card">

      <h2>Current Prediction</h2>

      <h1>{currentSign}</h1>

      <p>
        Confidence: {(confidence * 100).toFixed(1)}%
      </p>

    </div>
  );
}

export default PredictionBox;