import json
import os
import random
from datetime import datetime
from pathlib import Path
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Ensure tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Paths
ROOT = Path(__file__).parent
INTENTS_FILE = ROOT / "intents.json"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
VECT_FILE = MODEL_DIR / "vect.joblib"
MODEL_FILE = MODEL_DIR / "clf.joblib"
LE_FILE = MODEL_DIR / "le.joblib"

def load_intents(path=INTENTS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def prepare(intents):
    texts, labels, tag2responses = [], [], {}
    for intent in intents.get("intents", []):
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        for p in patterns:
            texts.append(p)
            labels.append(tag)
        tag2responses[tag] = responses or [f"Sorry, I don't have an answer for {tag}"]
    return texts, labels, tag2responses

class CantiliverChatbot:
    def __init__(self):
        self.vect = None
        self.clf = None
        self.le = None
        self.tag2responses = {}
        # Try to load model if exists
        if VECT_FILE.exists() and MODEL_FILE.exists() and LE_FILE.exists():
            self.vect = joblib.load(VECT_FILE)
            self.clf = joblib.load(MODEL_FILE)
            self.le = joblib.load(LE_FILE)

    def train(self, intents):
        texts, labels, tag2responses = prepare(intents)
        self.vect = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                token_pattern=r"(?u)\b\w+\b"   # keeps short words like "hi", "ok"
            )
        X = self.vect.fit_transform(texts)
        self.le = LabelEncoder().fit(labels)
        y = self.le.transform(labels)
        self.clf = LogisticRegression(max_iter=300).fit(X, y)
        self.tag2responses = tag2responses
        joblib.dump(self.vect, VECT_FILE)
        joblib.dump(self.clf, MODEL_FILE)
        joblib.dump(self.le, LE_FILE)

    def predict(self, text, threshold=0.15):
        X = self.vect.transform([text])
        probs = self.clf.predict_proba(X)[0]
        idx = probs.argmax()
        score = probs[idx]
        label = self.le.inverse_transform([idx])[0]
        if score < threshold:
            return "I didn't quite get that. Can you rephrase?"
        return random.choice(self.tag2responses.get(label, ["Sorry, I don't know that."]))

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def get_response(bot, user_text):
    return bot.predict(user_text)

if __name__ == "__main__":
    clear_screen()
    print("🤖 Cantiliver AI ready! Type 'exit' or 'quit' to stop.\n")

    intents = load_intents()
    bot = CantiliverChatbot()
    if bot.vect is None:
        print("Training model for the first time...")
        bot.train(intents)
        print("Training completed.\n")

    # Create a chat log file
    log_name = LOG_DIR / f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(log_name, "w", encoding="utf-8") as log:
        while True:
            user = input("You: ").strip()
            if user.lower() in ("exit", "quit"):
                print("Bot: Goodbye! 👋")
                break
            if not user:
                print("Bot: Please say something!")
                continue
            response = bot.predict(user)
            print(f"Bot: {response}")
            log.write(f"You: {user}\nBot: {response}\n")


