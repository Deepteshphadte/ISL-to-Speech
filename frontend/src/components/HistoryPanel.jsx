import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function HistoryPanel() {

  const { history } =
    useContext(PredictionContext);

  return (
    <div className="card">

      <h2>History</h2>

      {history.map((item, index) => (
        <p key={index}>{item}</p>
      ))}

    </div>
  );
}

export default HistoryPanel;