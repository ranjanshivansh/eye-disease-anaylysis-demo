import { useEffect, useMemo, useRef, useState } from "react";
import { Bars, GUIDE, MODEL_COLOR, MODEL_ORDER, Swatch, pct } from "../shared.jsx";

const FAMILIES = ["CNN baseline", "Foundation model", "Fusion", "Ours"];

export default function Analyze({ last, setLast }) {
  const [cat, setCat] = useState(null);
  const [sel, setSel] = useState([]);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [drag, setDrag] = useState(false);
  const [zoom, setZoom] = useState(null);
  const [tab, setTab] = useState(null);
  const [incImg, setIncImg] = useState(true);
  const [repBusy, setRepBusy] = useState(false);
  const inp = useRef();

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((d) => {
      setCat(d);
      const ok = d.models.filter((m) => m.available).map((m) => m.id);
      setSel(ok.includes("2c") ? ["2c"] : ok.slice(0, 1));
    }).catch(() => setErr("Could not reach the server."));
  }, []);

  const models = useMemo(() => (cat ? [...cat.models].sort((a, b) => MODEL_ORDER.indexOf(a.id) - MODEL_ORDER.indexOf(b.id)) : []), [cat]);
  const avail = models.filter((m) => m.available).map((m) => m.id);
  const toggle = (id) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  async function run(f = file, ids = sel) {
    if (!f) return;
    if (!ids.length) { setErr("Select at least one model first."); return; }
    setErr(""); setBusy(true); setLast(null);
    const fd = new FormData(); fd.append("file", f); fd.append("models", ids.join(","));
    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Server error " + r.status);
      const d = await r.json();
      setLast(d); setTab(d.models.find((m) => m.ok)?.id || null);
    } catch (e) { setErr(e.message); }
    setBusy(false);
  }
  async function report() {
    if (!last) return;
    setRepBusy(true); setErr("");
    try {
      const body = { original: last.original, filename: file ? file.name : "image", include_image: incImg, models: last.models.filter((m) => m.ok) };
      const r = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Server error " + r.status);
      const url = URL.createObjectURL(await r.blob());
      const a = document.createElement("a"); a.href = url; a.download = "retinal_report.pdf"; document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) { setErr(e.message); }
    setRepBusy(false);
  }
  function pick(f) { if (f) { setFile(f); run(f); } }

  const okModels = last?.models?.filter((m) => m.ok) || [];
  const cur = okModels.find((m) => m.id === tab) || okModels[0];
  const res = cur?.results;

  return (
    <div className="page">
      <section className="card">
        <div className="rowhead">
          <h2>1. Choose the models to run</h2>
          <div className="btnrow">
            <button className="ghost" onClick={() => setSel(avail)}>All available</button>
            <button className="ghost" onClick={() => setSel(avail.includes("2c") ? ["2c"] : [])}>Only 2c</button>
            <button className="ghost" onClick={() => setSel([])}>Clear</button>
          </div>
        </div>
        {!cat && <p className="muted">Loading models...</p>}
        {FAMILIES.map((fam) => {
          const list = models.filter((m) => m.family === fam);
          if (!list.length) return null;
          return (
            <div key={fam} className="fam">
              <div className="famname">{fam === "Ours" ? "Our model" : fam + (fam === "Fusion" ? " models (RETFound + ResNet50)" : "s")}</div>
              <div className="mgrid">
                {list.map((m) => (
                  <label key={m.id} className={"mopt " + (sel.includes(m.id) ? "on " : "") + (m.available ? "" : "off")} title={m.available ? m.description : "Checkpoint file missing: " + m.missing}>
                    <input type="checkbox" disabled={!m.available} checked={sel.includes(m.id)} onChange={() => toggle(m.id)} />
                    <Swatch id={m.id} />
                    <span className="mtext">
                      <b>{m.name}</b>
                      <small>{m.available ? m.description : "Not installed: add " + m.missing + " to backend/checkpoints"}</small>
                    </span>
                    {m.heatmaps && <span className="tag">heatmaps</span>}
                    {!m.available && <span className="tag warn">missing</span>}
                  </label>
                ))}
              </div>
            </div>
          );
        })}
        <p className="muted">
          {sel.length} selected. Several models are run one after the other on the same image, so more models take longer; each large model loads the first time it is used.
          Class-specific heatmaps are available for 2c only.
        </p>
      </section>

      <div
        className={"drop " + (drag ? "drag" : "")}
        onClick={() => inp.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
      >
        <input ref={inp} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files[0])} />
        <strong>2. Drop a fundus photograph here, or click to choose one</strong>
        <span>JPG or PNG. The image is analysed on this computer with the models selected above.</span>
      </div>
      {file && !busy && (
        <div className="btnrow">
          <button className="primary" onClick={() => run()}>Run the {sel.length} selected model{sel.length === 1 ? "" : "s"} again on {file.name}</button>
          {last && okModels.length > 0 && (
            <>
              <button className="ghost" onClick={report} disabled={repBusy}>{repBusy ? "Preparing PDF..." : "Download report (PDF)"}</button>
              <label className="inc"><input type="checkbox" checked={incImg} onChange={(e) => setIncImg(e.target.checked)} /> include the photo</label>
            </>
          )}
        </div>
      )}

      {busy && <div className="note busy"><span className="spin" /> Analysing with {sel.length} model{sel.length === 1 ? "" : "s"}. Large models need a few seconds each on CPU, and longer the first time they are loaded.</div>}
      {err && <div className="note err">{err}</div>}
      {last?.fake && <div className="note err">UI test mode: these numbers are random, not from the real models.</div>}
      {last?.models?.filter((m) => !m.ok).map((m) => <div className="note err" key={m.id}><b>{m.name}</b> could not run: {m.error}</div>)}

      {okModels.length > 1 && <Agreement models={okModels} />}

      {res && (
        <>
          {okModels.length > 1 && (
            <div className="tabs">
              {okModels.map((m) => (
                <button key={m.id} className={cur.id === m.id ? "on" : ""} onClick={() => setTab(m.id)}><Swatch id={m.id} /> {m.name}</button>
              ))}
            </div>
          )}
          <div className="grid2">
            <section className="card">
              <h2>Image</h2>
              <img className="orig" src={last.original} alt="uploaded fundus" />
              <p className="muted">All models took {last.seconds} s in total. {cur.name}: {cur.seconds} s.</p>
            </section>
            <section className="card">
              <h2>{cur.name}: predicted probabilities</h2>
              <Bars classes={res.map((r) => r.name)} probs={res.map((r) => r.prob)} thresholds={res.map((r) => r.threshold)} flags={res.map((r) => r.positive)} />
              <p className="muted">Thresholds: {cur.threshold_source}.</p>
            </section>
          </div>

          {cur.heatmaps ? (
            <section className="card">
              <h2>Where {cur.name} looked, per disease</h2>
              <p className="muted">Class-specific Grad-CAM. Warm colours mark regions that pushed the score of that disease up. Click a map to enlarge.</p>
              <div className="maps">
                {res.map((r) => (
                  <figure key={r.name} onClick={() => setZoom(r)}>
                    <img src={r.heatmap} alt={r.name + " heatmap"} />
                    <figcaption>
                      <b>{r.name}</b> <span className={"badge " + (r.positive ? "yes" : "no")}>{r.positive ? "Flagged" : "Not flagged"}</span>
                      <small>Expect: {GUIDE[r.name]}</small>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </section>
          ) : (
            <p className="muted">{cur.name} gives probabilities only. Select the 2c model to also see per-disease heatmaps.</p>
          )}
        </>
      )}

      {zoom && (
        <div className="modal" onClick={() => setZoom(null)}>
          <div onClick={(e) => e.stopPropagation()}>
            <img src={zoom.heatmap} alt="" />
            <p><b>{zoom.name}</b>: {GUIDE[zoom.name]}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function Agreement({ models }) {
  const classes = models[0].results.map((r) => r.name);
  return (
    <section className="card">
      <h2>Models side by side</h2>
      <div className="scroll">
        <table className="agree">
          <thead>
            <tr>
              <th>Disease</th>
              {models.map((m) => <th key={m.id}><span className="th"><Swatch id={m.id} /> {m.name}</span></th>)}
              <th>Flagged by</th>
            </tr>
          </thead>
          <tbody>
            {classes.map((c, j) => {
              const flags = models.map((m) => m.results[j].positive);
              const n = flags.filter(Boolean).length;
              return (
                <tr key={c}>
                  <td><b>{c}</b></td>
                  {models.map((m) => (
                    <td key={m.id} className={m.results[j].positive ? "flag" : ""}>
                      {pct(m.results[j].prob)} {m.results[j].positive && <small>flagged</small>}
                    </td>
                  ))}
                  <td>
                    <span className={"badge " + (n === models.length ? "yes" : n ? "mix" : "no")}>
                      {n === models.length ? "all " + n : n ? n + " of " + models.length + " (disagree)" : "none"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted">A cell is highlighted when that model's probability reaches its own tuned threshold. Different models can disagree on the same image; that is expected and is why the paper compares them on a large test set.</p>
    </section>
  );
}
