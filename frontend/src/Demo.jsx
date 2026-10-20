import React, { useEffect, useMemo, useRef, useState } from "react";

/* Example data only: nothing here comes from a real patient. */
const SAMPLE = {
  file: "sample_heart_recording.wav",
  seconds: 14,
  bpm: 72,
  quality: "good",
  cycles: 17,
  probs: [
    ["murmur", 0.71],
    ["normal", 0.18],
    ["other abnormal", 0.08],
    ["aortic stenosis", 0.02],
    ["mitral regurgitation", 0.01],
  ],
  sources: [
    { title: "Heart murmur (general)", score: 0.62, text: "A murmur is an extra sound from turbulent blood flow. Timing, loudness and location help tell harmless from significant murmurs." },
    { title: "Normal heart sounds", score: 0.31, text: "Innocent flow murmurs are soft, short, and common in children and during fever, pregnancy or exercise." },
    { title: "When to seek urgent care", score: 0.2, text: "Chest pain, fainting or severe breathlessness need in-person care whatever a screening tool says." },
  ],
  report: {
    urgency: "soon",
    summary: "Your recording looks similar to a heart murmur, an extra whooshing sound between heartbeats. Many murmurs are harmless, but a doctor should check it.",
    explanation: "A murmur happens when blood flows in a way that makes extra noise. In children and some adults it is harmless. Sometimes it points to a problem with a heart valve, so it is worth confirming.",
    steps: [
      "Book a visit with your doctor in the next few weeks and share this report.",
      "Ask whether an echocardiogram (an ultrasound scan of the heart) is needed.",
      "Note any breathlessness, chest pain or dizziness, and tell the doctor.",
      "Get urgent help right away if you faint, have chest pain, or struggle to breathe.",
    ],
    cites: ["Heart murmur (general)", "When to seek urgent care"],
  },
};

const STEPS = [
  { id: "upload", title: "Upload", where: "Your browser to POST /analyze", text: "You pick or drop a heart sound recording. The server checks it is audio, between 3 and 60 seconds, and under 15 MB. It is read in memory and never saved." },
  { id: "measure", title: "Measure", where: "app/signal", text: "The sound is cleaned with a 25 to 400 Hz filter, then its loudness envelope shows each S1 ('lub') and S2 ('dub'). Heart rate comes from how often the full cycle repeats." },
  { id: "spectrogram", title: "Picture", where: "app/model", text: "The recording is cut into 4-second windows. Each becomes a spectrogram, a small image of pitch over time that the classifier can read like a photo." },
  { id: "classify", title: "Classify", where: "ONNX CNN", text: "A small convolutional network scores each window against 8 conditions. The window scores are averaged into one probability per condition." },
  { id: "retrieve", title: "Look up", where: "app/rag", text: "The top conditions are used to search a library of cardiology notes. The best passages are passed on so the report is grounded in sources." },
  { id: "report", title: "Report", where: "LangGraph + Gemini", text: "An agent checks quality and confidence, writes a plain-language report from the scores and the retrieved sources, then a safety step adds urgency and the disclaimer." },
];

/* deterministic pseudo-random so the example looks the same every time */
function rng(seed) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5;
}

function makeSignal() {
  const N = 900, cycle = 190, r = rng(11), out = [];
  const burst = (d, w, a) => a * Math.exp(-(d * d) / (2 * w * w)) * Math.sin(d * 0.8);
  for (let i = 0; i < N; i++) {
    const p = (i + 30) % cycle;
    // a soft murmur between S1 and S2 (systole)
    const murmur = p > 50 && p < 125 ? 0.16 * r() * 2 : 0;
    out.push(burst(p, 36, 1) + burst(p, 112, 0.7) + murmur + r() * 0.06);
  }
  return out;
}

function Wave({ withEnvelope, withMarks }) {
  const sig = useMemo(makeSignal, []);
  const W = 900, H = 150, mid = H / 2;
  const env = useMemo(() => sig.map((_, i) => {
    let m = 0; for (let k = -12; k <= 12; k++) { const q = sig[Math.min(sig.length - 1, Math.max(0, i + k))]; m += q * q; }
    return Math.sqrt(m / 25);
  }), [sig]);
  const pts = (arr, k) => arr.map((v, i) => `${i},${(mid - v * k).toFixed(1)}`).join(" ");
  const marks = [];
  for (let c = -30; c < sig.length; c += 190) {
    [[36, "S1"], [112, "S2"]].forEach(([o, t]) => { const x = c + o; if (x > 0 && x < sig.length - 10) marks.push([x, t]); });
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="dwave" role="img" aria-label="Example heart sound waveform">
      <line x1="0" x2={W} y1={mid} y2={mid} stroke="currentColor" opacity=".12" />
      {withMarks && marks.map(([x, t], i) => (
        <g key={i}><line x1={x} x2={x} y1="10" y2={H - 6} className="mark2" /><text x={x + 4} y="20" className="lbl2">{t}</text></g>
      ))}
      <polyline points={pts(sig, 42)} className="sound2" />
      {withEnvelope && <polyline points={pts(env, 60)} className="env2" />}
    </svg>
  );
}

function Spectrogram() {
  const ref = useRef();
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d"), W = c.width, H = c.height, r = rng(5);
    const css = getComputedStyle(document.documentElement);
    const hot = css.getPropertyValue("--accent").trim() || "#c8102e";
    ctx.fillStyle = css.getPropertyValue("--card").trim() || "#fff"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = hot;
    const cols = 150, rows = 32, cw = W / cols, rh = H / rows;
    for (let x = 0; x < cols; x++) {
      const p = (x * 6 + 30) % 190;
      const s1 = Math.exp(-((p - 36) ** 2) / 90), s2 = Math.exp(-((p - 112) ** 2) / 70);
      const murmur = p > 50 && p < 125 ? 0.25 : 0;
      for (let y = 0; y < rows; y++) {
        const f = y / rows; // 0 = low pitch (bottom)
        const a = (s1 + s2 * 0.8) * Math.exp(-f * 4) + murmur * Math.exp(-((f - 0.35) ** 2) / 0.05) * (0.6 + r());
        ctx.globalAlpha = Math.max(0, Math.min(1, a + r() * 0.06));
        ctx.fillRect(x * cw, H - (y + 1) * rh, cw + 0.5, rh + 0.5);
      }
    }
    ctx.globalAlpha = 1;
  }, []);
  return (
    <div className="spec">
      <canvas ref={ref} width="900" height="170" aria-label="Example spectrogram: pitch over time" />
      <div className="wins">
        {[0, 1, 2].map((i) => <span key={i} style={{ left: `${i * 29}%`, width: "38%" }}>window {i + 1}</span>)}
      </div>
      <div className="axis"><span>time →</span><span>↑ pitch (up to 800 Hz)</span></div>
    </div>
  );
}

function Visual({ id }) {
  if (id === "upload")
    return (
      <div className="upl">
        <div className="file"><b>{SAMPLE.file}</b><span>{SAMPLE.seconds} s · 4 kHz · 112 KB</span></div>
        <ul className="checks">
          <li>Readable audio file</li><li>At least 3 seconds long</li><li>Under 15 MB</li><li>Decoded in memory, never stored</li>
        </ul>
      </div>
    );
  if (id === "measure")
    return (
      <>
        <Wave withEnvelope withMarks />
        <div className="tiles">
          <div><b>{SAMPLE.bpm}</b><span>beats per minute</span></div>
          <div><b>{SAMPLE.cycles}</b><span>heart cycles found</span></div>
          <div><b>{SAMPLE.quality}</b><span>signal quality</span></div>
        </div>
      </>
    );
  if (id === "spectrogram") return <Spectrogram />;
  if (id === "classify")
    return (
      <div className="bars">
        {SAMPLE.probs.map(([l, p], i) => (
          <div className="bar" key={l}>
            <span>{l}</span>
            <div><i style={{ width: `${p * 100}%`, animationDelay: `${i * 80}ms` }} /></div>
            <b>{Math.round(p * 100)}%</b>
          </div>
        ))}
        <p className="fine">Example scores. Murmur is the top result, so murmur notes are looked up next.</p>
      </div>
    );
  if (id === "retrieve")
    return (
      <div className="srcs">
        {SAMPLE.sources.map((s) => (
          <div className="src" key={s.title}>
            <div className="srch"><b>{s.title}</b><span className="mono">match {s.score.toFixed(2)}</span></div>
            <p>{s.text}</p>
          </div>
        ))}
      </div>
    );
  const r = SAMPLE.report;
  return (
    <div className="final">
      <ol className="nodes2">
        {["interpret", "retrieve", "write report", "safety check"].map((n) => <li key={n}>{n}</li>)}
      </ol>
      <div className="card2">
        <span className="badge2">{r.urgency}</span>
        <h4>{r.summary}</h4>
        <h5>What this means</h5><p>{r.explanation}</p>
        <h5>Next steps</h5><ol>{r.steps.map((s) => <li key={s}>{s}</li>)}</ol>
        <h5>Sources</h5><p className="fine">{r.cites.join(" · ")}</p>
        <p className="fine">This is an AI screening aid, not a medical diagnosis. It cannot replace an examination by a clinician.</p>
      </div>
    </div>
  );
}

export default function Demo() {
  // "#/demo/3" opens step 3 directly, so a step can be linked or shared
  const start = Math.min(STEPS.length, Math.max(1, parseInt(window.location.hash.split("/")[2], 10) || 1)) - 1;
  const [i, setI] = useState(start);
  const [playing, setPlaying] = useState(false);
  const reduce = useMemo(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches, []);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setI((x) => { if (x >= STEPS.length - 1) { setPlaying(false); return x; } return x + 1; }), 4200);
    return () => clearInterval(t);
  }, [playing]);

  const step = STEPS[i];
  return (
    <>
      <header>
        <h1>How it works</h1>
        <p className="muted">A guided walk through one example recording, from upload to the final report. All numbers and text on this page are example data, not a real patient.</p>
      </header>

      <nav className="stepper" aria-label="Pipeline steps">
        {STEPS.map((s, k) => (
          <button key={s.id} className={k === i ? "on" : k < i ? "done" : ""} onClick={() => { setI(k); setPlaying(false); }} aria-current={k === i ? "step" : undefined}>
            <span className="num">{k + 1}</span><span>{s.title}</span>
          </button>
        ))}
      </nav>

      <section className="card stage" key={step.id}>
        <div className="stagehead">
          <div><h2>{i + 1}. {step.title}</h2><span className="where mono">{step.where}</span></div>
        </div>
        <p className="why">{step.text}</p>
        <div className={reduce ? "vis" : "vis fade"}><Visual id={step.id} /></div>
      </section>

      <div className="ctrls">
        <button onClick={() => { setI(Math.max(0, i - 1)); setPlaying(false); }} disabled={i === 0}>Back</button>
        <button onClick={() => { setPlaying(false); setI(Math.min(STEPS.length - 1, i + 1)); }} disabled={i === STEPS.length - 1}>Next</button>
        <button className="primary" onClick={() => { if (i >= STEPS.length - 1) setI(0); setPlaying((p) => !p); }}>{playing ? "Pause" : "Play walkthrough"}</button>
        <a className="try" href="#/">Try it with your own recording</a>
      </div>
    </>
  );
}
