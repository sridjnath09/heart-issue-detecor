import io
import os

import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agent.graph import build_graph
from .model.classifier import HeartClassifier
from .signal.analyzer import analyze_signal

load_dotenv()
MAX_BYTES = 15 * 1024 * 1024
MAX_SECONDS = 60

app = FastAPI(title="Heart Issue Detector")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGIN", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])
classifier = HeartClassifier()
graph = build_graph()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier.available, "labels": classifier.labels}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB)")
    try:  # audio is decoded in memory and never written to disk
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True, duration=MAX_SECONDS)
    except Exception:
        raise HTTPException(400, "Could not decode audio. Upload a WAV/MP3/FLAC file.")
    if len(y) < sr * 3:
        raise HTTPException(400, "Recording too short (need at least 3 seconds).")
    metrics = analyze_signal(y, sr)
    prediction = classifier.predict(y, sr) if classifier.available else None
    out = graph.invoke({"metrics": metrics, "prediction": prediction})
    return {"metrics": metrics, "prediction": prediction, "report": out["report"]}
