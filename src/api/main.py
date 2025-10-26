from datasets import load_dataset
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from text_classifier import Classifier

from .models import UserInputText

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

training_ds = load_dataset("ucirvine/sms_spam", split="train")

label_map = {0: "legit", 1: "spam", -1: "unknown"}

docs = []

for entry in training_ds:
    if isinstance(entry, dict):
        doc = entry.get("sms", "")
        label = label_map[entry.get("label", -1)]
        docs.append((doc, label))

c = Classifier()

c.train(docs)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/categories")
def categories():
    return [label for k, label in label_map.items() if k >= 0]


@app.post("/predict/")
async def predict(data: UserInputText):
    text = data.text
    return c.predict(text)
