# 🖐️ Real-Time Indian Sign Language to Speech Converter

An AI-powered web application that recognizes **Indian Sign Language (ISL)** gestures in real time and converts them into natural text and speech. The system leverages **Computer Vision**, **Deep Learning**, **Natural Language Processing**, and **Speech Synthesis** to bridge the communication gap between hearing-impaired individuals and people unfamiliar with sign language.

---

## 📌 Features

- 🎥 Real-time webcam-based gesture recognition
- ✋ Hand landmark detection using MediaPipe
- 🤖 Gesture recognition using LSTM with Transformer Encoder
- 📝 AI-powered sentence refinement using Google Gemini
- 🔊 Offline speech synthesis using pyttsx3
- 🌐 React + FastAPI web application
- 📊 Live prediction confidence
- 📜 Sentence formation and history tracking
- 🖐️ Supports detection of both hands

---

# 🏗️ System Architecture

```
Webcam
      │
      ▼
OpenCV Video Capture
      │
      ▼
MediaPipe Hands
(21 landmarks per hand)
      │
      ▼
Landmark Preprocessing
      │
      ▼
LSTM + Transformer Encoder
      │
      ▼
Gesture Prediction
      │
      ▼
Sentence Formation
      │
      ▼
Google Gemini
(Natural Language Refinement)
      │
      ▼
pyttsx3
(Text-to-Speech)
      │
      ▼
Speech Output
```

---

# 🛠️ Tech Stack

## Frontend

- React
- Vite
- JavaScript
- CSS

## Backend

- FastAPI
- Python

## AI / Machine Learning

- PyTorch
- LSTM
- Transformer Encoder
- MediaPipe Hands
- OpenCV

## NLP

- Google Gemini API

## Speech

- pyttsx3

---

# 📂 Project Structure

```
ISL-to-Speech
│
├── backend
│   ├── main.py
│   ├── predictor.py
│   ├── gemini_service.py
│   ├── speech_engine.py
│   └── model_loader.py
│
├── frontend
│   ├── src
│   ├── public
│   ├── package.json
│   └── vite.config.js
│
├── models
│   ├── lstm_model.py
│   └── lstm_model.pth
│
├── utils
│   └── preprocessing.py
│
├── data
│   ├── processed
│   ├── labels.npy
│   └── raw
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/ISL-to-Speech.git

cd ISL-to-Speech
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Frontend Dependencies

```bash
cd frontend

npm install
```

---

# ▶️ Running the Project

## Start Backend

```bash
uvicorn backend.main:app --reload
```

Backend runs on

```
http://127.0.0.1:8000
```

---

## Start Frontend

```bash
cd frontend

npm run dev
```

Frontend runs on

```
http://localhost:5173
```

---

# ⚙️ Workflow

1. User performs ISL gesture.
2. Webcam captures live video.
3. MediaPipe extracts hand landmarks.
4. Landmark sequences are preprocessed.
5. LSTM + Transformer Encoder predicts gesture.
6. Recognized gestures form a sentence.
7. Google Gemini refines the sentence.
8. pyttsx3 converts text into speech.

---

# 📊 Model Information

| Parameter | Value |
|------------|-------|
| Framework | PyTorch |
| Model | LSTM + Transformer Encoder |
| Sequence Length | 30 Frames |
| Features per Frame | 126 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

---

# 📈 Applications

- Real-time communication
- Educational institutions
- Healthcare services
- Public service accessibility
- Workplace communication
- Assistive technology

---

# 🔮 Future Scope

- Larger ISL dataset
- Continuous sign language recognition
- Mobile application
- Multilingual speech output
- Cloud deployment
- Improved contextual understanding

---

# 👩‍💻 Authors

**Meha Dangui**

Final Year B.E. Computer Engineering Student

Don Bosco College of Engineering, Goa

---

# 📚 Technologies Used

- Python
- React
- FastAPI
- OpenCV
- MediaPipe
- PyTorch
- Google Gemini API
- pyttsx3
- JavaScript
- HTML
- CSS

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!

---

# 📄 License

This project is developed for educational and research purposes.
