import pyttsx3


class SpeechEngine:

    def __init__(self):
        self.rate = 160

    def speak(self, text):

        if not text:
            return

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()