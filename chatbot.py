import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

# Only download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Use correct file name (change if your file name is different)
filename = "intents.json"  # or "inntents.json" if that's your actual filename

# Load intents data with error handling
try:
    with open(filename, encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: {filename} is not valid JSON.")
    exit(1)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Use NLTK tokenizer with token_pattern=None
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None)
X = vectorizer.fit_transform(sentences)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

def get_response(message, threshold=0.5):
    X_test = vectorizer.transform([message])
    probs = model.predict_proba(X_test)[0]
    max_prob = max(probs)
    if max_prob < threshold:
        return "I'm not sure I understand. Can you rephrase?"
    prediction = model.predict(X_test)[0]
    predicted_tag = label_encoder.inverse_transform([prediction])[0]
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't get that."

if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Bot: Please enter a message.")
            continue
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)
