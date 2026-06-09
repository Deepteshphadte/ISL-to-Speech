import { createContext, useState } from "react";

export const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {

  const [currentSign, setCurrentSign] = useState("No Sign");
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState("");
  const [history, setHistory] = useState([]);

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
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
};