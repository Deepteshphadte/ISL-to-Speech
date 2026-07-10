import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function HistoryPanel() {

  const { history } =
    useContext(PredictionContext);

  const uniqueHistory = [
    ...new Set(
      history.slice(-15)
    )
  ];

  return (

    <div className="card">

      <h2>Recent Predictions</h2>

      <div className="history-chips">

        {uniqueHistory.map(
          (item, index) => (

            <span
              key={index}
              className="history-chip"
            >
              {item}
            </span>

          )
        )}

      </div>

    </div>

  );
}

export default HistoryPanel;