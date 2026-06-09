import { useContext, useEffect } from "react";
import Header from "../components/Header";
import WebcamFeed from "../components/WebcamFeed";
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

function Home() {
    const {
  setCurrentSign,
  setConfidence,
  setSentence,
  setHistory
} = useContext(PredictionContext);
useEffect(() => {

  const ws = connectWebSocket((data) => {

    setCurrentSign(data.sign);

    setConfidence(data.confidence);

    setSentence(data.sentence);

    setHistory((prev) => [
      ...prev,
      data.sign
    ]);

  });


  return () => ws.close();

}, []);
const sendFrame = (frame) => {

  const socket = getSocket();

  if (
    socket &&
    socket.readyState === WebSocket.OPEN
  ) {

    socket.send(frame);

  }

};

const speak = (text) => {

  const utterance =
      new SpeechSynthesisUtterance(text);

  window.speechSynthesis.speak(
      utterance
  );
};
  return (
    <div className="container">
      <Header />

      <div className="dashboard">

        <div className="left-panel">
          <WebcamFeed sendFrame={sendFrame} />
        </div>

        <div className="right-panel">
          <PredictionBox />
          <SentenceBox />
          <Controls speak={speak} />
          <HistoryPanel />
        </div>

      </div>
    </div>
  );
}

export default Home;