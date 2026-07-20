import { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";
import API from "../api/backend";

function Controls() {

  const {
    refinedSentence,
    setSentence,
    setHistory,
    autoSpeak,
    setAutoSpeak
  } = useContext(PredictionContext);

  const copySentence = async () => {

    try {

      await navigator.clipboard.writeText(
        refinedSentence
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
      [refinedSentence],
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

const toggleSpeech = async () => {

    try {

        const response =
            await API.post("/toggle_speech");

        setAutoSpeak(
            response.data.auto_speak
        );

    }
    catch (error) {

        console.log(error);

    }

};

  return (

    <div className="card controls-card">


      <button
          className="toggle-btn"
          onClick={toggleSpeech}
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