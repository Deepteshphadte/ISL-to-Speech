import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";
import API from "../api/backend";

function Controls({ speak }) {

  const {
    sentence,
    setSentence,
    setHistory,
    autoSpeak,
    setAutoSpeak
  } = useContext(PredictionContext);

  const copySentence = async () => {

    try {

      await navigator.clipboard.writeText(
        sentence
      );

      alert(
        "Sentence copied!"
      );

    } catch (error) {

      console.log(error);

    }

  };

  const exportSentence = () => {

    const blob = new Blob(
      [sentence],
      {
        type: "text/plain"
      }
    );

    const url =
      window.URL.createObjectURL(
        blob
      );

    const link =
      document.createElement("a");

    link.href = url;

    link.download =
      "sentence.txt";

    link.click();

    window.URL.revokeObjectURL(
      url
    );
  };
  const clearAll = async () => {

    try {

      await API.post("/clear");

      setSentence("");
      setHistory([]);

    } catch (error) {

      console.log(error);

    }

  };

  return (

    <div className="card controls-card">

      <button
        className="speak-btn"
        onClick={() =>
          speak(sentence)
        }
      >
        🔊 Speak
      </button>

      <button
        className="toggle-btn"
        onClick={() =>{
          setAutoSpeak(!autoSpeak);
          console.log("AutoSpeak:", !autoSpeak);
        }}
      >
        {autoSpeak
          ? "🔊 Auto Speak ON"
          : "🔇 Auto Speak OFF"}
      </button>

      <button
        className="clear-btn"
        onClick={clearAll}
      >
        🗑 Clear Sentence
      </button>

      <button
        className="export-btn"
        onClick={exportSentence}
      >
        ⬇ Export
      </button>

      <button
        className="copy-btn"
        onClick={copySentence}
      >
        📋 Copy
      </button>

    </div>

  );
}

export default Controls;