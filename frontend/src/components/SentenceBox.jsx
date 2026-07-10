import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function SentenceBox() {

  const { sentence } =
    useContext(PredictionContext);

  return (
    <div className="card">

      <h2>Recognized Sentence</h2>

      <textarea
        rows="6"
        value={sentence}
        readOnly
      />

    </div>
  );
}

export default SentenceBox;