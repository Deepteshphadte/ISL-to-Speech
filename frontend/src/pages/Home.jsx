import { useContext, useEffect, useRef } from "react";
import Header from "../components/Header";
//import WebcamFeed from "../components/WebcamFeed";
import PredictionBox from "../components/PredictionBox";
import SentenceBox from "../components/SentenceBox";
import HistoryPanel from "../components/HistoryPanel";
import Controls from "../components/Controls";
import API from "../api/backend";
import {
  connectWebSocket,
  getSocket
} from "../services/websocket";
import { PredictionContext } from "../context/PredictionContext";
import "../styles/main.css";
import ConnectionStatus
from "../components/ConnectionStatus";
import StatusCard
from "../components/StatusCard";
import WebcamFeedMP from "../components/WebcamFeedMP";

function Home() {
    const {
  setCurrentSign,
  setConfidence,
  setSentence,
  setRefinedSentence,
  setHistory,
  autoSpeak,
  setConnectionStatus,
  setLandmarks
} = useContext(PredictionContext);


const processingRef = useRef(false);


useEffect(() => {

  const ws = connectWebSocket(

    (data) => {

      processingRef.current = false;

      setCurrentSign(data.sign);

      setConfidence(data.confidence);

      setSentence(data.sentence);

      setRefinedSentence(data.refined_sentence || "");

      setLandmarks(data.landmarks);
      

      setHistory((prev) => {

        if (
          prev[prev.length - 1]
          === data.sign
        ) {
          return prev;
        }

        return [
          ...prev,
          data.sign
        ];
      });
    },

    () => {
      setConnectionStatus(
        "Connected"
      );
    },

    () => {
      setConnectionStatus(
        "Disconnected"
      );
    }
  );


  return () => ws.close();

}, []);
const sendFrame = (frame) => {

   console.log("sendFrame called");

  if (processingRef.current) return;

  const socket = getSocket();

   console.log(socket);

  if (
    socket &&
    socket.readyState === WebSocket.OPEN
  ) {
        console.log("Sending frame to backend");

        processingRef.current = true;

        socket.send(frame);

    } else {

        console.log("Socket not open");

    }
};

  return (
    <div className="container">
      <Header />

      <div className="dashboard">

        <div className="left-panel">
          <WebcamFeedMP sendFrame={sendFrame}/>
        </div>

        <div className="right-panel">
          <ConnectionStatus />
          <StatusCard />
          <PredictionBox />
          <SentenceBox />
          <Controls />
          <HistoryPanel />
        </div>

      </div>
    </div>
  );
}

export default Home;