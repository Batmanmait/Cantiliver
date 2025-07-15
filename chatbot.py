import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

sentences = []
labels = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    classes.append(intent["tag"])

vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(sentences)
y = labels

model = LogisticRegression()
model.fit(X, y)

def get_response(message):
    X_test = vectorizer.transform([message])
    prediction = model.predict(X_test)[0]
    for intent in data["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

