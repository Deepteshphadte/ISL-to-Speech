import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function ConnectionStatus() {

    const { connectionStatus } =
        useContext(PredictionContext);

    return (

        <div className="status-card">

            <h3>Connection</h3>

            <div
                className={
                    connectionStatus === "Connected"
                    ? "connected"
                    : "disconnected"
                }
            >
                {connectionStatus}
            </div>

        </div>

    );
}

export default ConnectionStatus;