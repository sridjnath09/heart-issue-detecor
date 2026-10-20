import React, { useEffect, useRef, useState } from "react";
import Demo from "./Demo.jsx";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const pretty = (s) => s.replace(/_/g, " ");

function Wave({ metrics }) {
  const { waveform: w, envelope: e, peak_idx: p } = metrics;
  const H = 120, W = 600, max = Math.max(...w.map(Math.abs), 1e-6);
  const pt = (arr, i) => `${(i / (arr.length - 1)) * W},${H / 2 - (arr[i] / max) * (H / 2 - 4)}`;
  const line = (arr) => arr.map((_, i) => pt(arr, i)).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="wave" role="img" aria-label="Filtered heart sound waveform">
      <polyline points={line(w)} fill="none" stroke="var(--accent)" strokeWidth="1" />
      <polyline points={line(e)} fill="none" stroke="var(--muted)" strokeWidth="1" opacity=".6" />
      {p.map((i) => <line key={i} x1={(i / (w.length - 1)) * W} x2={(i / (w.length - 1)) * W} y1="0" y2={H} stroke="var(--warn)" opacity=".35" />)}
    </svg>
  );
}

function Analyzer() {
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [name, setName] = useState("");
  const input = useRef();

  async function submit(file) {
    setBusy(true); setErr(""); setRes(null); setName(file.name);
    try {
      const fd = new FormData(); fd.append("file", file);
      const r = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Server error ${r.status}`);
      setRes(await r.json());
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  const rep = res?.report;
  return (
    <>
      <header>
        <h1>Heart Sound Analyzer</h1>
        <p className="muted">Upload a heart sound recording (WAV/MP3/FLAC, 3–60 s) for an AI screening report.</p>
        <a className="demolink" href="#/demo">See how it works: a step-by-step demo →</a>
      </header>

      <div className="drop" onClick={() => input.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); e.dataTransfer.files[0] && submit(e.dataTransfer.files[0]); }}>
        <input ref={input} type="file" accept="audio/*" hidden onChange={(e) => e.target.files[0] && submit(e.target.files[0])} />
        {busy ? "Analyzing " + name + "…" : "Click or drop an audio file here"}
      </div>
      {err && <div className="card err">{err}</div>}

      {res && (
        <>
          <section className={`card urgency ${rep.urgency}`}>
            <span className="badge">{rep.urgency}</span>
            <h2>{rep.summary}</h2>
            {rep.confidence_note && <p className="muted">{rep.confidence_note}</p>}
          </section>

          {res.prediction && (
            <section className="card">
              <h3>Condition probabilities</h3>
              {res.prediction.predictions.slice(0, 5).map((p) => (
                <div className="bar" key={p.label}>
                  <span>{pretty(p.label)}</span>
                  <div><i style={{ width: `${p.probability * 100}%` }} /></div>
                  <b>{(p.probability * 100).toFixed(0)}%</b>
                </div>
              ))}
            </section>
          )}

          <section className="card">
            <h3>Signal: {res.metrics.heart_rate_bpm ? `about ${Math.round(res.metrics.heart_rate_bpm)} beats per minute` : "heart rate unclear"}, quality {res.metrics.quality}</h3>
            <Wave metrics={res.metrics} />
          </section>

          <section className="card">
            <h3>Findings</h3><ul>{rep.findings.map((f, i) => <li key={i}>{f}</li>)}</ul>
            <h3>What this means</h3><p>{rep.explanation}</p>
            <h3>Recommended next steps</h3><ol>{rep.next_steps.map((s, i) => <li key={i}>{s}</li>)}</ol>
            {rep.citations?.length > 0 && <><h3>Sources</h3><ul className="muted">{rep.citations.map((c, i) => <li key={i}>{c}</li>)}</ul></>}
            <button onClick={() => window.print()}>Print / save as PDF</button>
          </section>
          <p className="disclaimer">{rep.disclaimer}</p>
        </>
      )}
    </>
  );
}

export default function App() {
  const [route, setRoute] = useState(window.location.hash);
  useEffect(() => {
    const on = () => { setRoute(window.location.hash); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  if (route.startsWith("#/demo")) return <Demo route={route} />;
  return (
    <main>
      <nav className="top">
        <a href="#/" className="cur">Analyzer</a>
        <a href="#/demo">How it works</a>
      </nav>
      <Analyzer />
    </main>
  );
}
