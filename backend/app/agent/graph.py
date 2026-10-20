"""LangGraph agent: interpret -> retrieve -> write report -> safety check.

The LLM only explains and recommends from (a) the model's output, (b) signal metrics and
(c) retrieved knowledge. It is told not to diagnose. If no API key is configured, a
deterministic template report is produced so the app still works.
"""
import logging
import os
import re
import time
from typing import Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph

from ..rag.retriever import Retriever

log = logging.getLogger("heart.agent")

DISCLAIMER = ("This is an AI screening aid, not a medical diagnosis. It cannot replace an "
              "examination by a qualified clinician. If you have chest pain, fainting or severe "
              "breathlessness, seek urgent care.")
LLM_BUDGET_S = 36
MODEL_TIMEOUT_S = 12  # Google rejects deadlines under 10 s
LOW_CONF = 0.55
# Classes with little or single-source training data: reported, but flagged as less reliable.
EXPERIMENTAL = {"aortic_stenosis", "mitral_regurgitation", "mitral_stenosis", "mitral_valve_prolapse", "extrasystole"}
DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite"

_retriever: Optional[Retriever] = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


class Report(BaseModel):
    summary: str = Field(description="2-3 sentence plain-language summary")
    findings: list[str] = Field(description="Key observations from the model and signal metrics")
    explanation: str = Field(description="Plain-language explanation of what the top condition means")
    next_steps: list[str] = Field(description="Ordered, concrete recommended next steps")
    urgency: str = Field(description="one of: routine, soon, urgent")
    confidence_note: str = ""
    citations: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    metrics: dict
    prediction: Optional[dict]
    context: list[dict]
    report: dict
    flags: list[str]


def interpret(state: State) -> State:
    m, p, flags = state["metrics"], state.get("prediction"), []
    if m.get("quality") == "poor":
        flags.append("poor_quality")
    if p is None:
        flags.append("no_model")
    else:
        if p["top_probability"] < LOW_CONF:
            flags.append("low_confidence")
        if p["top_label"] in EXPERIMENTAL:
            flags.append("experimental_class")
    return {"flags": flags}


def retrieve(state: State) -> State:
    r, p, flags = retriever(), state.get("prediction"), state["flags"]
    ctx: list[dict] = []
    if p:
        for item in p["predictions"][:2]:
            if item["probability"] >= 0.15:
                ctx += r.by_doc(item["label"]) or r.search(item["label"].replace("_", " "), 2)
    if "poor_quality" in flags:
        ctx += r.by_doc("poor_quality")
    ctx += r.by_doc("when_to_seek_urgent_care")
    seen, uniq = set(), []
    for c in ctx:
        if c["id"] not in seen:
            seen.add(c["id"]); uniq.append(c)
    return {"context": uniq}


def _plain(text: str, n: int = 3) -> str:
    """First few sentences of a knowledge chunk, minus the 'Source:' line."""
    body = " ".join(l for l in text.splitlines() if not l.startswith("Source:"))
    return " ".join(re.split(r"(?<=[.;])\s+", body)[:n]).strip()


def _template_report(state: State) -> Report:
    p, flags, ctx = state.get("prediction"), state["flags"], state["context"]
    cites = sorted({f"{c['title']} - {c['source']}" for c in ctx})
    steps = ["Discuss this result with a doctor; share the recording and this report."]
    if p:
        top = p["top_label"].replace("_", " ")
        findings = [f"{x['label'].replace('_', ' ')}: {x['probability']:.0%}" for x in p["predictions"][:3]]
        summary = f"The recording most resembles '{top}' ({p['top_probability']:.0%} model confidence)."
        urgency = "routine" if p["top_label"] == "normal" else "soon"
        if p["top_label"] != "normal":
            steps += ["Ask for an echocardiogram and ECG to confirm or exclude a heart condition."]
    else:
        findings, urgency = ["Classification model not loaded; only signal metrics are available."], "routine"
        summary = "Signal analysis completed; automated condition classification is unavailable."
    if "experimental_class" in flags:
        steps.append("Treat this result as tentative: the app has seen few examples of this condition.")
    if "poor_quality" in flags:
        steps.insert(0, "Re-record in a quiet room with the sensor on bare skin (10-20 s, breath held).")
    return Report(summary=summary, findings=findings,
                  explanation=_plain(ctx[0]["text"]) if ctx else "",
                  next_steps=steps, urgency=urgency, citations=cites,
                  confidence_note="Low confidence - treat as inconclusive." if "low_confidence" in flags else "")


def write_report(state: State) -> State:
    if not os.getenv("GOOGLE_API_KEY"):
        return {"report": _template_report(state).model_dump()}
    from langchain_google_genai import ChatGoogleGenerativeAI
    kb = "\n\n".join(f"[{c['title']} | {c['source']}]\n{c['text']}" for c in state["context"])
    m = {k: v for k, v in state["metrics"].items() if k not in ("waveform", "envelope", "peak_idx")}
    prompt = (
        "You write the result page of a heart-sound screening app for an ordinary person with no medical "
        "training. Speak directly to the reader as 'you' and 'your recording', never as 'the patient'. "
        "Use short sentences and everyday words; explain any medical term the first time you use it "
        "(for example: 'echocardiogram, an ultrasound scan of the heart'). Stay calm and neutral: no alarm "
        "unless the sources show real red flags.\n"
        "Rules:\n"
        "- Use ONLY the model output, signal metrics and knowledge excerpts below. Do not add facts.\n"
        "- Never state a diagnosis. Say 'may suggest' or 'looks similar to'.\n"
        "- summary: 2 sentences max. explanation: 3-4 short sentences on what the top result means, "
        "rewritten in your own plain words (do not copy the excerpt).\n"
        "- findings: 2-4 bullets in plain words, no raw jargon like 'SNR' unless you explain it.\n"
        "- next_steps: 3-5 concrete actions in priority order, each one sentence.\n"
        "- If flags include poor_quality, no_model or low_confidence, say the result is unclear and make "
        "'record again' the first step. Explain how: quiet room, sensor on bare skin, hold breath, 10-20 seconds.\n"
        "- If flags include experimental_class, say plainly that the app has seen few examples of this "
        "condition, so this result is less reliable and needs confirming by a doctor.\n"
        "- urgency: 'urgent' only if the sources show red flags, otherwise 'routine' or 'soon'.\n"
        "- Heart rate: quote heart_rate_bpm only if it is not null, as 'about N beats per minute'. Treat the "
        "beat-to-beat variability numbers (hrv, rhythm) as rough; do not draw conclusions from them.\n"
        "- citations: the titles of the excerpts you used.\n\n"
        f"FLAGS: {state['flags']}\nMODEL OUTPUT: {state.get('prediction')}\nSIGNAL METRICS: {m}\n\n"
        f"KNOWLEDGE:\n{kb}")
    # Try each model in turn (newest first). One attempt each, and a hard 36 s budget overall,
    # so an overloaded Gemini can never stall the app; the template report is the last resort.
    deadline = time.monotonic() + LLM_BUDGET_S
    for name in [x.strip() for x in os.getenv("GEMINI_MODELS", DEFAULT_MODELS).split(",") if x.strip()]:
        left = deadline - time.monotonic()
        if left < 10:  # not enough time for another full attempt
            break
        try:
            llm = ChatGoogleGenerativeAI(model=name, max_output_tokens=1500, max_retries=0,
                                         timeout=MODEL_TIMEOUT_S).with_structured_output(Report)
            rep = llm.invoke(prompt)
            if rep:
                return {"report": rep.model_dump()}
        except Exception as e:
            log.warning("model %s failed: %s: %s", name, type(e).__name__, str(e)[:120])
    log.warning("all models failed, using template report")
    return {"report": _template_report(state).model_dump()}


def safety_check(state: State) -> State:
    rep, flags = dict(state["report"]), state["flags"]
    if rep.get("urgency") not in ("routine", "soon", "urgent"):
        rep["urgency"] = "soon"
    if flags and rep["urgency"] == "routine" and "low_confidence" in flags:
        rep["urgency"] = "soon"
    rep["disclaimer"] = DISCLAIMER
    rep["flags"] = flags
    return {"report": rep}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("interpret", interpret), ("retrieve", retrieve),
                     ("write_report", write_report), ("safety_check", safety_check)]:
        g.add_node(name, fn)
    g.set_entry_point("interpret")
    g.add_edge("interpret", "retrieve"); g.add_edge("retrieve", "write_report")
    g.add_edge("write_report", "safety_check"); g.add_edge("safety_check", END)
    return g.compile()
