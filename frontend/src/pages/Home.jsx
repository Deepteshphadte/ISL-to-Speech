import { useContext, useEffect, useRef } from "react";
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
import ConnectionStatus
from "../components/ConnectionStatus";
import StatusCard
from "../components/StatusCard";

function Home() {
    const {
  setCurrentSign,
  setConfidence,
  setSentence,
  setHistory,
  autoSpeak,
  setConnectionStatus,
  setLandmarks
} = useContext(PredictionContext);

const lastSentenceRef = useRef("");
const autoSpeakRef = useRef(autoSpeak);

useEffect(() => {

  autoSpeakRef.current = autoSpeak;

}, [autoSpeak]);

useEffect(() => {

  const ws = connectWebSocket(

    (data) => {

      setCurrentSign(data.sign);

      setConfidence(data.confidence);

      setSentence(data.sentence);
      setLandmarks(data.landmarks);
      if (
        autoSpeakRef.current &&
        data.sentence &&
        data.sentence !== lastSentenceRef.current
      ) {

        const utterance =
          new SpeechSynthesisUtterance(
            data.sentence
          );

        window.speechSynthesis.speak(
          utterance
        );

        lastSentenceRef.current =
          data.sentence;
      }

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
          <ConnectionStatus />
          <StatusCard />
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