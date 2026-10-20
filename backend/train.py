"""Train the multi-condition heart-sound CNN and export ONNX (run on Colab/GPU, not in deploy image).

Data: a folder per class under DATA_DIR, each holding .wav files, named after labels in models/labels.json:
    DATA_DIR/normal/*.wav  DATA_DIR/murmur/*.wav  DATA_DIR/aortic_stenosis/*.wav ...
Build it from CirCor 2022 (murmur present/absent), PhysioNet/CinC 2016 (normal/abnormal -> other_abnormal)
and the Kaggle heart-sound valve set (AS/MR/MS/MVP). Split by PATIENT, not by file, to avoid leakage.

    pip install torch librosa scikit-learn onnx
    python train.py --data DATA_DIR --epochs 30
"""
import argparse, json
from pathlib import Path

import librosa, numpy as np, torch, torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

from app.model.classifier import FRAMES, N_MELS, SEG_S, SR, log_mel, segments

ROOT = Path(__file__).parent


def _one(f, li):
    audio, _ = librosa.load(f, sr=SR, mono=True)
    audio /= np.max(np.abs(audio)) + 1e-9
    return [(log_mel(seg)[:, :FRAMES], li) for seg in segments(audio)]


def windows(files, labels):
    """Feature extraction is CPU-bound; spread files across all cores."""
    out = Parallel(n_jobs=-1, verbose=5)(delayed(_one)(f, li) for f, li in files)
    pairs = [p for chunk in out for p in chunk]
    return np.stack([x for x, _ in pairs])[:, None].astype(np.float32), np.array([y for _, y in pairs])


def net(n):
    def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 64), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.4), nn.Linear(64, n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    data = Path(a.data)
    labels = [l for l in json.loads((ROOT / "models/labels.json").read_text())
              if (data / l).exists() and any((data / l).glob("*.wav"))]
    print("training classes:", labels)
    files = [(f, i) for i, l in enumerate(labels) for f in sorted((data / l).glob("*.wav"))]
    # split by RECORDING before windowing so windows of one file never straddle train/test
    # group = patient id for CirCor (circor_<pid>_<loc>.wav); every other file is its own group
    group = lambda f: f.name.split("_")[1] if f.name.startswith("circor_") else f.name
    gi = [group(f) for f, _ in files]
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(files, [i for _, i in files], gi))
    ftr, fte = [files[i] for i in tr], [files[i] for i in te]
    Xtr, ytr = windows(ftr, labels); Xte, yte = windows(fte, labels)
    w = torch.tensor(len(ytr) / (len(labels) * np.maximum(np.bincount(ytr, minlength=len(labels)), 1)), dtype=torch.float32)
    m, opt = net(len(labels)), None
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]; opt.zero_grad()
            loss_fn(m(Xt[idx]), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pred = m(torch.tensor(Xte)).argmax(1).numpy()
        print(f"epoch {ep + 1}: val acc {(pred == yte).mean():.3f}")
    print(classification_report(yte, pred, labels=range(len(labels)), target_names=labels, zero_division=0))
    torch.onnx.export(m, torch.zeros(1, 1, N_MELS, FRAMES), ROOT / "models/heart_model.onnx",
                      input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}})
    (ROOT / "models/labels.json").write_text(json.dumps(labels))
    print("saved models/heart_model.onnx + labels.json")
