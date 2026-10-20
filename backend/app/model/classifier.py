"""ONNX multi-condition heart-sound classifier.

Expects models/heart_model.onnx (input: log-mel patch [1,1,64,128], output: logits)
and models/labels.json (list of class names). If absent, `available` is False and the
API returns signal metrics only -- it never invents a prediction.
"""
import json
import os
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
SR, N_MELS, FRAMES, HOP = 4000, 64, 128, 32  # ~1.0 s per patch (128*32/4000 ≈ 1.02 s)
SEG_S = 4.0


def log_mel(y: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=256, hop_length=HOP, n_mels=N_MELS, fmax=800)
    m = librosa.power_to_db(m, ref=np.max)
    return (m - m.mean()) / (m.std() + 1e-6)


def segments(y: np.ndarray):
    """Split recording into 4 s windows (pad short audio); model sees one log-mel per window."""
    n = int(SEG_S * SR)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    for i in range(0, max(1, len(y) - n + 1), n // 2):
        yield y[i:i + n]


class HeartClassifier:
    def __init__(self):
        self.session = None
        self.labels: list[str] = []
        onnx, lab = MODEL_DIR / "heart_model.onnx", MODEL_DIR / "labels.json"
        if onnx.exists() and lab.exists():
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self.labels = json.loads(lab.read_text())

    @property
    def available(self) -> bool:
        return self.session is not None

    def predict(self, y: np.ndarray, sr: int) -> dict:
        import librosa
        if sr != SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        y = y / (np.max(np.abs(y)) + 1e-9)
        batch = np.stack([log_mel(s)[:, :FRAMES] for s in segments(y)])[:, None].astype(np.float32)
        logits = self.session.run(None, {self.session.get_inputs()[0].name: batch})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)  # average over windows
        order = np.argsort(-probs)
        return {
            "predictions": [{"label": self.labels[i], "probability": round(float(probs[i]), 4)} for i in order],
            "top_label": self.labels[order[0]],
            "top_probability": round(float(probs[order[0]]), 4),
            "n_windows": int(len(batch)),
        }
