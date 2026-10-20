import React, { useEffect, useMemo, useRef, useState } from "react";
import "./demo.css";

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

const STAGES = [
  {
    id: "upload", n: 1, title: "Upload", color: "#4f8ef7", where: "POST /analyze · FastAPI",
    tag: "Check the file, keep nothing",
    text: "You pick or drop a heart sound recording. The server checks that it is real audio, between 3 and 60 seconds, and under 15 MB. It is decoded in memory and never written to disk.",
    analogy: "Like a clinic front desk: it confirms the paperwork is usable before anyone is seen, and shreds it afterwards.",
    facts: ["WAV, MP3 or FLAC", "Max 15 MB, 3 to 60 seconds", "Decoded in memory, not stored", "Clear error messages for bad files"],
    chips: ["FastAPI", "librosa", "soundfile"],
  },
  {
    id: "measure", n: 2, title: "Measure", color: "#10b981", where: "app/signal",
    tag: "Find the beats, grade the signal",
    text: "A 25 to 400 Hz filter removes rumble and hiss. The loudness envelope then shows each S1 ('lub') and S2 ('dub'). Heart rate comes from how often the full cycle repeats, so the two sounds are not counted as two beats.",
    analogy: "Like a stethoscope user tuning out the room noise to listen for the rhythm first.",
    facts: ["Band-pass filter 25 to 400 Hz", "Cycle length from autocorrelation", "Signal quality: good, fair or poor", "Heart rate shown only when reliable"],
    chips: ["SciPy", "NumPy"],
  },
  {
    id: "spectrogram", n: 3, title: "Picture", color: "#06b6d4", where: "app/model · log-mel",
    tag: "Turn sound into an image",
    text: "The recording is cut into 4-second windows. Each becomes a spectrogram, a small image of pitch over time, which the classifier can read like a photo.",
    analogy: "Like sheet music: time runs left to right, pitch runs up, and brighter marks are louder.",
    facts: ["4-second windows, 50% overlap", "64 mel bands up to 800 Hz", "128 time frames per window", "Normalised per window"],
    chips: ["log-mel", "librosa"],
  },
  {
    id: "classify", n: 4, title: "Classify", color: "#7c3aed", where: "ONNX CNN · onnxruntime",
    tag: "Score eight conditions",
    text: "A small convolutional neural network scores every window against 8 conditions. The window scores are averaged into one probability per condition.",
    analogy: "Like a panel of reviewers who each look at one slice of the recording, then average their votes.",
    facts: ["8 classes, trained on about 6,450 recordings", "Overall accuracy 73.5% on held-out data", "Valve classes marked experimental", "Model file is only 268 KB"],
    chips: ["PyTorch", "ONNX", "CNN"],
  },
  {
    id: "retrieve", n: 5, title: "Look up", color: "#f59e0b", where: "app/rag · retrieval",
    tag: "Ground the answer in sources",
    text: "The top conditions are used to search a small library of cardiology notes. The best passages travel with the result, so the report can cite where its facts come from.",
    analogy: "Like a doctor pulling the right page from a reference book before explaining a result.",
    facts: ["10 curated knowledge notes", "TF-IDF search over chunks", "Always adds 'when to seek urgent care'", "Every passage keeps its source"],
    chips: ["RAG", "scikit-learn"],
  },
  {
    id: "report", n: 6, title: "Report", color: "#f43f5e", where: "app/agent · LangGraph + Gemini",
    tag: "Explain it and recommend next steps",
    text: "An agent checks quality and confidence, then Gemini writes a plain-language report from the scores and the retrieved sources. A safety step sets the urgency and adds the disclaimer. If Gemini is busy, a template report is used.",
    analogy: "Like a nurse who translates the test result into everyday words and tells you what to do next.",
    facts: ["Four nodes: interpret, retrieve, write, safety check", "Up to three Gemini models tried, 36 s limit", "Never states a diagnosis", "Tentative wording for experimental classes"],
    chips: ["LangGraph", "Gemini", "Pydantic"],
  },
];

const EXTRA = [
  { id: "safety", title: "Safety and limits", color: "#f43f5e", tag: "What the app can and cannot do" },
  { id: "deploy", title: "Deployment", color: "#4f8ef7", tag: "Where each part runs" },
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
    const murmur = p > 50 && p < 125 ? 0.16 * r() * 2 : 0; // soft murmur between S1 and S2
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
    ctx.fillStyle = "#0e1526"; ctx.fillRect(0, 0, W, H);
    const cols = 150, rows = 32, cw = W / cols, rh = H / rows;
    for (let x = 0; x < cols; x++) {
      const p = (x * 6 + 30) % 190;
      const s1 = Math.exp(-((p - 36) ** 2) / 90), s2 = Math.exp(-((p - 112) ** 2) / 70);
      const murmur = p > 50 && p < 125 ? 0.25 : 0;
      for (let y = 0; y < rows; y++) {
        const f = y / rows;
        const a = (s1 + s2 * 0.8) * Math.exp(-f * 4) + murmur * Math.exp(-((f - 0.35) ** 2) / 0.05) * (0.6 + r());
        ctx.globalAlpha = Math.max(0, Math.min(1, a + r() * 0.06));
        ctx.fillStyle = f > 0.5 ? "#f59e0b" : "#06b6d4";
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

function ReportCard() {
  const r = SAMPLE.report;
  return (
    <div className="card2">
      <span className="badge2">{r.urgency}</span>
      <h4>{r.summary}</h4>
      <h5>What this means</h5><p>{r.explanation}</p>
      <h5>Next steps</h5><ol>{r.steps.map((s) => <li key={s}>{s}</li>)}</ol>
      <h5>Sources</h5><p className="fine">{r.cites.join(" · ")}</p>
      <p className="fine">This is an AI screening aid, not a medical diagnosis. It cannot replace an examination by a clinician.</p>
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
  return (
    <div className="final">
      <ol className="nodes2">
        {["interpret", "retrieve", "write report", "safety check"].map((n) => <li key={n}>{n}</li>)}
      </ol>
      <ReportCard />
    </div>
  );
}

/* ---------- pages ---------- */

function PipelineStrip({ active, done }) {
  return (
    <div className="strip">
      <div className="striplabel">End-to-end pipeline · one example recording</div>
      <div className="stripflow">
        {STAGES.map((s, k) => (
          <React.Fragment key={s.id}>
            <a href={`#/demo/stage/${s.n}`} className={`pill ${active === k ? "act" : ""} ${done > k ? "dn" : ""}`} style={{ "--c": s.color }}>{s.title}</a>
            {k < STAGES.length - 1 && <span className="arr">→</span>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function Home() {
  const cards = [
    { href: "#/demo/live", num: "▶", title: "Live demo: watch it run", desc: "An animated run of one example recording through all six stages, ending in the final report.", color: "#f43f5e" },
    ...STAGES.map((s) => ({ href: `#/demo/stage/${s.n}`, num: String(s.n).padStart(2, "0"), title: s.title, desc: s.tag + ".", color: s.color })),
    ...EXTRA.map((e) => ({ href: `#/demo/${e.id}`, num: "··", title: e.title, desc: e.tag + ".", color: e.color })),
  ];
  return (
    <>
      <section className="hero">
        <div className="eyebrow">Heart sound analysis · screening aid</div>
        <h1>Heart Sound Analyzer <span>from recording to readable report</span></h1>
        <p>Upload a heart sound recording. The app measures the signal, scores it against eight conditions, looks up cardiology references, and writes a plain-language report with next steps. This walkthrough follows one example recording through every stage.</p>
        <div className="chips">
          {["FastAPI", "SciPy", "CNN + ONNX", "RAG", "LangGraph", "Gemini", "React", "Netlify + Render"].map((c, i) => <span key={c} className="chip" style={{ "--c": ["#4f8ef7", "#10b981", "#7c3aed", "#f59e0b", "#f43f5e", "#06b6d4", "#4f8ef7", "#10b981"][i] }}>{c}</span>)}
        </div>
      </section>
      <PipelineStrip active={-1} done={0} />
      <div className="sectionlabel">{cards.length} topics · click any card to open</div>
      <div className="cardgrid">
        {cards.map((c) => (
          <a key={c.title} href={c.href} className="tcard" style={{ "--c": c.color }}>
            <span className="num">{c.num}</span><b>{c.title}</b><p>{c.desc}</p><span className="go">→</span>
          </a>
        ))}
      </div>
    </>
  );
}

function StagePage({ n }) {
  const s = STAGES[n - 1];
  if (!s) return <p>Stage not found. <a href="#/demo">Back to home</a></p>;
  const prev = STAGES[n - 2], next = STAGES[n];
  return (
    <>
      <a className="back" href="#/demo">← Back to home</a>
      <section className="stagehero" style={{ "--c": s.color }}>
        <div className="eyebrow" style={{ color: s.color }}>Stage {s.n} of {STAGES.length} · {s.where}</div>
        <h1>{s.title}: <span style={{ color: s.color }}>{s.tag}</span></h1>
        <p>{s.text}</p>
        <div className="chips">{s.chips.map((c) => <span key={c} className="chip" style={{ "--c": s.color }}>{c}</span>)}</div>
      </section>
      <PipelineStrip active={n - 1} done={n - 1} />
      <div className="panel2" style={{ "--c": s.color }}>
        <div className="paneltitle">What you would see</div>
        <Visual id={s.id} />
        <p className="fine example">Example data, not a real patient.</p>
      </div>
      <div className="two2">
        <div className="infobox"><div className="paneltitle">Analogy</div><p>{s.analogy}</p></div>
        <div className="infobox"><div className="paneltitle">Under the hood</div><ul>{s.facts.map((f) => <li key={f}>{f}</li>)}</ul></div>
      </div>
      <div className="pager">
        {prev ? <a href={`#/demo/stage/${prev.n}`}>← {prev.title}</a> : <span />}
        {next ? <a href={`#/demo/stage/${next.n}`}>{next.title} →</a> : <a href="#/demo/live">Watch it run →</a>}
      </div>
    </>
  );
}

const EVENTS = [
  [0, "POST /analyze  sample_heart_recording.wav (112 KB)"],
  [0, "decoded 14.0 s of audio at 4000 Hz, in memory"],
  [0, "checks passed: readable, 3 to 60 s, under 15 MB"],
  [1, "band-pass filter 25 to 400 Hz applied"],
  [1, "found 34 heart sounds in 17 cycles"],
  [1, "heart rate about 72 beats per minute"],
  [1, "signal quality: good"],
  [2, "cut into 5 windows of 4 s with 50% overlap"],
  [2, "built a 64 x 128 spectrogram for each window"],
  [3, "CNN scored 5 windows against 8 conditions"],
  [3, "averaged: murmur 0.71 · normal 0.18 · other abnormal 0.08"],
  [4, "searched 10 cardiology notes for 'murmur'"],
  [4, "top match: Heart murmur (general), score 0.62"],
  [4, "added: When to seek urgent care"],
  [5, "interpret: confidence 71%, quality good, no flags"],
  [5, "write report: Gemini returned a structured report"],
  [5, "safety check: urgency set to 'soon', disclaimer added"],
  [5, "done. Report ready."],
];

function Live() {
  const [idx, setIdx] = useState(0);        // number of log lines shown
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const logRef = useRef();

  useEffect(() => {
    if (!playing) return;
    if (idx >= EVENTS.length) { setPlaying(false); return; }
    const t = setTimeout(() => setIdx((i) => i + 1), 900 / speed);
    return () => clearTimeout(t);
  }, [playing, idx, speed]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [idx]);

  const cur = idx === 0 ? -1 : EVENTS[idx - 1][0];
  const finished = idx >= EVENTS.length;
  const play = () => { if (finished) setIdx(0); setPlaying(true); };
  return (
    <>
      <a className="back" href="#/demo">← Back to home</a>
      <section className="stagehero" style={{ "--c": "#f43f5e" }}>
        <div className="eyebrow" style={{ color: "#f43f5e" }}>Live demo · example data</div>
        <h1>Watch one recording <span style={{ color: "#f43f5e" }}>go through the pipeline</span></h1>
        <p>Press play. Each line is something the server does for a real upload, shown here with example numbers.</p>
      </section>
      <PipelineStrip active={cur} done={cur} />
      <div className="controls">
        <button className="primary" onClick={play}>{playing ? "Playing…" : finished ? "Replay" : idx ? "Resume" : "Play demo"}</button>
        <button onClick={() => setPlaying(false)} disabled={!playing}>Pause</button>
        <button onClick={() => { setPlaying(false); setIdx(0); }} disabled={idx === 0}>Reset</button>
        <span className="speed">Speed
          {[1, 2, 4].map((s) => <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s}x</button>)}
        </span>
      </div>
      <div className="livegrid">
        <div className="panel2" style={{ "--c": cur >= 0 ? STAGES[cur].color : "#243350" }}>
          <div className="paneltitle">{cur >= 0 ? `Stage ${cur + 1}: ${STAGES[cur].title}` : "Waiting to start"}</div>
          {cur >= 0 ? <Visual key={cur} id={STAGES[cur].id} /> : <p className="fine">Press "Play demo" to start.</p>}
        </div>
        <div className="logwrap">
          <div className="paneltitle">Server log</div>
          <div className="log" ref={logRef} aria-live="polite">
            {EVENTS.slice(0, idx).map(([s, t], k) => (
              <div key={k} className="ln" style={{ "--c": STAGES[s].color }}>
                <span className="ts">{String(k * 0.3).slice(0, 3).padStart(3, "0")}s</span>
                <span className="st">{STAGES[s].title}</span><span>{t}</span>
              </div>
            ))}
            {idx === 0 && <div className="ln muted">no activity yet</div>}
          </div>
        </div>
      </div>
      {finished && (
        <div className="panel2" style={{ "--c": "#f43f5e" }}>
          <div className="paneltitle">Final report shown to the user</div>
          <ReportCard />
        </div>
      )}
    </>
  );
}

function Safety() {
  const rows = [
    ["Not a diagnosis", "Every report carries a disclaimer and uses wording like 'looks similar to'. A clinician must confirm anything the app suggests."],
    ["Experimental classes", "Aortic stenosis, mitral regurgitation, mitral stenosis, mitral valve prolapse and extrasystole had few, short training clips. When one is the top result, the report says it is tentative."],
    ["Known weak spots", "Overall accuracy is 73.5%. Only 66% of normal windows are recognised as normal, so some healthy hearts get flagged. Murmur calls are right about 43% of the time."],
    ["Poor recordings", "If the signal is noisy or the model is unsure, the report says the result is unclear and 'record again' comes first, with tips for a better recording."],
    ["Privacy", "Audio is decoded in memory and never saved. The report text is generated from scores and metrics, not from the audio itself."],
    ["Urgent symptoms", "Chest pain, fainting or severe breathlessness always need in-person care, whatever the app says."],
  ];
  return (
    <>
      <a className="back" href="#/demo">← Back to home</a>
      <section className="stagehero" style={{ "--c": "#f43f5e" }}>
        <div className="eyebrow" style={{ color: "#f43f5e" }}>Cross-cutting</div>
        <h1>Safety and limits: <span style={{ color: "#f43f5e" }}>what the app can and cannot do</span></h1>
      </section>
      <div className="cardgrid">{rows.map(([t, d]) => <div key={t} className="tcard static" style={{ "--c": "#f43f5e" }}><b>{t}</b><p>{d}</p></div>)}</div>
    </>
  );
}

function Deploy() {
  const boxes = [
    ["Your browser", "Opens the page and uploads audio", "#4f8ef7"],
    ["Netlify", "Serves the React app (static files)", "#10b981"],
    ["Render", "Runs the Python API in a Docker container", "#f59e0b"],
    ["Gemini API", "Writes the plain-language report", "#7c3aed"],
  ];
  return (
    <>
      <a className="back" href="#/demo">← Back to home</a>
      <section className="stagehero" style={{ "--c": "#4f8ef7" }}>
        <div className="eyebrow" style={{ color: "#4f8ef7" }}>Cross-cutting</div>
        <h1>Deployment: <span style={{ color: "#4f8ef7" }}>where each part runs</span></h1>
        <p>Netlify only serves static files, so the Python model, retrieval and agent run on a separate server.</p>
      </section>
      <div className="stripflow big">
        {boxes.map(([t, d, c], k) => (
          <React.Fragment key={t}>
            <div className="flowbox" style={{ "--c": c }}><b>{t}</b><span>{d}</span></div>
            {k < boxes.length - 1 && <span className="arr">→</span>}
          </React.Fragment>
        ))}
      </div>
      <div className="two2">
        <div className="infobox"><div className="paneltitle">Free-tier behaviour</div><ul><li>The API sleeps when idle, so the first request can take about a minute.</li><li>Memory is limited to 512 MB, which is enough for this small model.</li><li>Report generation has a 36 second limit, then a template report is used.</li></ul></div>
        <div className="infobox"><div className="paneltitle">Configuration</div><ul><li>GOOGLE_API_KEY lives only on the server.</li><li>ALLOWED_ORIGIN limits which site can call the API.</li><li>VITE_API_URL points the app at the API.</li></ul></div>
      </div>
    </>
  );
}

/* ---------- shell ---------- */

export default function Demo({ route }) {
  const [open, setOpen] = useState(false);
  const parts = route.replace(/^#\/demo\/?/, "").split("/").filter(Boolean);
  const page = parts[0] || "home";
  useEffect(() => setOpen(false), [route]);

  const link = (href, label, color, key) => {
    const on = route === href || (href === "#/demo" && route === "#/demo/");
    return (
      <a key={key || href} href={href} className={`navlink ${on ? "on" : ""}`}>
        {color && <i style={{ background: color }} />}{label}
      </a>
    );
  };
  let body;
  if (page === "stage") body = <StagePage n={parseInt(parts[1], 10)} />;
  else if (page === "live") body = <Live />;
  else if (page === "safety") body = <Safety />;
  else if (page === "deploy") body = <Deploy />;
  else body = <Home />;

  return (
    <div className="xp">
      <header className="topbar">
        <button className="ham" onClick={() => setOpen(!open)} aria-label="Toggle navigation" aria-expanded={open}>☰</button>
        <b className="brand">Heart Sound <span>Analyzer</span></b>
        <a className="toapp" href="#/">Try the analyzer →</a>
      </header>
      <div className="body">
        <nav className={`side ${open ? "open" : ""}`} aria-label="Demo navigation">
          <div className="navhead">Navigation</div>
          {link("#/demo", "Home")}
          {link("#/demo/live", "Live demo")}
          <div className="navhead">Pipeline stages</div>
          {STAGES.map((s) => link(`#/demo/stage/${s.n}`, `Stage ${s.n}: ${s.title}`, s.color))}
          <div className="navhead">Cross-cutting</div>
          {EXTRA.map((e) => link(`#/demo/${e.id}`, e.title, e.color))}
        </nav>
        <main className="content">{body}</main>
      </div>
    </div>
  );
}
