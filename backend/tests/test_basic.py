import numpy as np
from fastapi.testclient import TestClient

from app.agent.graph import build_graph
from app.main import app
from app.rag.retriever import Retriever
from app.signal.analyzer import analyze_signal


def synth(sr=2000, secs=10, bpm=70):
    t = np.arange(sr * secs) / sr
    y = 0.01 * np.random.randn(len(t))
    for b in np.arange(0.2, secs, 60 / bpm):
        for off, f in ((0, 60), (0.3, 90)):
            m = (t > b + off) & (t < b + off + 0.08)
            y[m] += np.sin(2 * np.pi * f * t[m])
    return y.astype(np.float32), sr


def test_retriever_cited():
    r = Retriever().search("systolic murmur radiating to carotids", 3)
    assert r and r[0]["doc"] == "aortic_stenosis" and r[0]["source"]


def test_signal_detects_beats():
    y, sr = synth()
    m = analyze_signal(y, sr)
    assert 8 <= m["n_beats_detected"] <= 30 and m["quality"] in ("good", "fair")


def test_agent_template_without_model(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    y, sr = synth()
    out = build_graph().invoke({"metrics": analyze_signal(y, sr), "prediction": None})
    assert "no_model" in out["report"]["flags"] and out["report"]["disclaimer"]


def test_agent_with_prediction(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pred = {"predictions": [{"label": "aortic_stenosis", "probability": 0.8}, {"label": "normal", "probability": 0.2}],
            "top_label": "aortic_stenosis", "top_probability": 0.8, "n_windows": 3}
    y, sr = synth()
    out = build_graph().invoke({"metrics": analyze_signal(y, sr), "prediction": pred})
    assert out["report"]["urgency"] == "soon" and out["report"]["citations"]


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_heart_rate_not_doubled():
    for bpm in (50, 70, 100, 140):
        y, sr = synth(secs=15, bpm=bpm)
        m = analyze_signal(y, sr)
        assert abs(m["heart_rate_bpm"] - bpm) <= 3, (bpm, m["heart_rate_bpm"])


def test_experimental_class_flagged(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pred = {"predictions": [{"label": "mitral_stenosis", "probability": 0.9}], "top_label": "mitral_stenosis",
            "top_probability": 0.9, "n_windows": 1}
    y, sr = synth()
    out = build_graph().invoke({"metrics": analyze_signal(y, sr), "prediction": pred})["report"]
    assert "experimental_class" in out["flags"] and any("tentative" in s for s in out["next_steps"])


def test_non_normal_never_routine(monkeypatch):
    from app.agent import graph as g
    state = {"metrics": {}, "prediction": {"top_label": "murmur", "top_probability": 0.9, "predictions": []},
             "report": {"urgency": "routine"}, "flags": []}
    assert g.safety_check(state)["report"]["urgency"] == "soon"
