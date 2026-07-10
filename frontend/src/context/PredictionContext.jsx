import { createContext, useState } from "react";

export const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {

  const [currentSign, setCurrentSign] = useState("No Sign");

  const [confidence, setConfidence] = useState(0);

  const [sentence, setSentence] = useState("");

  const [history, setHistory] = useState([]);

  const [landmarks, setLandmarks] = useState([]);

  const [connectionStatus, setConnectionStatus] =
    useState("Disconnected");

  const [autoSpeak, setAutoSpeak] =
  useState(true);


  return (
    <PredictionContext.Provider
      value={{

        currentSign,
        setCurrentSign,

        confidence,
        setConfidence,

        sentence,
        setSentence,

        history,
        setHistory,

        connectionStatus,
        setConnectionStatus,

        autoSpeak,
        setAutoSpeak,

        landmarks,
        setLandmarks

      }}
    >
      {children}
    </PredictionContext.Provider>
  );
};