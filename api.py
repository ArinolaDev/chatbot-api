from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import numpy as np
import json
import random
import pickle
import nltk

from tensorflow.keras.models import model_from_json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# NLTK setup
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("index.html", "r") as f:
        return f.read()

# ✅ Load trained model and data
# Load architecture
with open("model_architecture.json", "r") as f:
    model_json = f.read()

model = model_from_json(model_json)

# Load weights
model.load_weights("model.weights.h5")

with open('intent.json') as file:
    intents = json.load(file)

with open("words.pkl", "rb") as f:
    words = pickle.load(f)

with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# Input format
class Message(BaseModel):
    text: str

# Preprocessing (MUST match training)
def bag_of_words(text):
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalnum()]
    return np.array([1 if w in tokens else 0 for w in words])

# Predict
def predict_class(text):
    bow = bag_of_words(text)
    res = model.predict(np.array([bow]), verbose=0)[0]
    idx = np.argmax(res)
    return classes[idx], res[idx]

# Response
def get_response(user_input):
    intent, confidence = predict_class(user_input)

    if confidence > 0.5:
        for i in intents['intents']:
            if i['tag'] == intent:
                return random.choice(i['responses'])

    return "Sorry, I didn't understand that."

# API endpoint
@app.post("/chat")
def chat(message: Message):
    response = get_response(message.text)
    return {"response": response}