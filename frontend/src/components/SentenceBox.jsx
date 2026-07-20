import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function SentenceBox() {

  const {
    sentence,
    refinedSentence
  } = useContext(PredictionContext);

  return (

  <div className="card">

      <h2>Recognized Sentence</h2>

      <textarea
          rows="2"
          value={sentence}
          readOnly
      />

      <h2
          style={{
              marginTop:20
          }}
      >
          AI Refined Sentence
      </h2>

      <textarea
          rows="3"
          value={refinedSentence}
          readOnly
      />

  </div>

  );
}

export default SentenceBox;