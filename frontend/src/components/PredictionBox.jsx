import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function PredictionBox() {

  const {
    currentSign,
    confidence
  } = useContext(PredictionContext);

  const percentage = Math.round(
    confidence * 100
  );

  const getBarColor = () => {

    if (percentage >= 85)
      return "#00ff66";

    if (percentage >= 60)
      return "#ffcc00";

    return "#ff4444";
  };

  return (

    <div className="card prediction-card">

      <h2>Current Sign</h2>

      <div className="prediction-sign">

        {currentSign}

      </div>

      <div className="confidence-section">

        <div className="confidence-label">

          Confidence

        </div>

        <div className="progress-bar">

          <div
            className="progress-fill"
            style={{
              width: `${percentage}%`,
              backgroundColor:
                getBarColor()
            }}
          />

        </div>

        <div className="confidence-text">

          {percentage}%

        </div>

      </div>

    </div>
  );
}

export default PredictionBox;