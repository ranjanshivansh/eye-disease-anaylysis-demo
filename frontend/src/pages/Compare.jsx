import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { MODEL_COLOR, MODEL_MARK, MODEL_ORDER, Mark, Swatch } from "../shared.jsx";

/* ------------------------------------------------------------------ metric catalogue (every number comes from results.json = the paper) */
const SCOPES = [["pooled", "Pooled test set"], ["D1", "D1 ODIR-5K"], ["D2", "D2 MuReD"], ["D3", "D3 RFMiD"]];
const rep = (k) => (m) => (m.report ? m.report[k] : null);
const METRICS = [
  { k: "auroc", label: "Macro-AUROC", scoped: true, get: (m, s) => (s === "pooled" ? m.pooled.auroc : m.source[s].auroc), fmt: (v) => v.toFixed(4), group: "Ranking quality", best: "high" },
  { k: "f1", label: "Macro-F1", scoped: true, get: (m, s) => (s === "pooled" ? m.pooled.f1 : m.source[s].f1), fmt: (v) => v.toFixed(4), group: "Operating point", best: "high" },
  { k: "ap", label: "PR-AUC (macro AP)", get: (m) => m.pooled.ap, fmt: (v) => v.toFixed(4), group: "Ranking quality", best: "high" },
  { k: "accuracy", label: "Accuracy (per-label)", get: rep("accuracy"), fmt: (v) => v.toFixed(2) + " %", group: "Operating point", best: "high" },
  { k: "exact_match", label: "Exact-match accuracy", get: rep("exact_match"), fmt: (v) => v.toFixed(2) + " %", group: "Operating point", best: "high" },
  { k: "precision", label: "Precision", get: rep("precision"), fmt: (v) => v.toFixed(2) + " %", group: "Operating point", best: "high" },
  { k: "recall", label: "Recall (sensitivity)", get: rep("recall"), fmt: (v) => v.toFixed(2) + " %", group: "Operating point", best: "high" },
  { k: "specificity", label: "Specificity", get: rep("specificity"), fmt: (v) => v.toFixed(2) + " %", group: "Operating point", best: "high" },
  { k: "latency_ms", label: "Latency (ms / image)", get: rep("latency_ms"), fmt: (v) => v.toFixed(2) + " ms", group: "Cost", best: "low" },
  { k: "fps", label: "Throughput (images / s)", get: rep("fps"), fmt: (v) => v.toFixed(2), group: "Cost", best: "high" },
  { k: "trainable_m", label: "Trainable parameters (M)", get: (m) => m.trainable_m, fmt: (v) => v.toFixed(1) + " M", group: "Cost", best: "low" },
  { k: "params_m", label: "Total parameters (M)", get: (m) => m.params_m, fmt: (v) => v.toFixed(1) + " M", group: "Cost", best: "low" },
];
const VIEWS = [["rank", "Ranking"], ["source", "By data source"], ["class", "By disease"], ["trade", "Trade-off"], ["paired", "Paired tests"]];

/* ------------------------------------------------------------------ helpers */
function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(720);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    setW(Math.max(300, Math.floor(el.getBoundingClientRect().width)));
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.floor(e.contentRect.width))));
    ro.observe(el); return () => ro.disconnect();
  }, []);
  return [ref, w];
}
function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw)); const f = raw / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p; }
function ticksFor(lo, hi, n = 5) {
  const step = niceStep((hi - lo) / n); const a = Math.floor(lo / step) * step; const out = [];
  for (let v = a; ; v += step) { out.push(+v.toFixed(10)); if (v >= hi - step * 1e-6) break; }
  return out;
}
const decimals = (step) => Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));

function Tip({ tip }) {
  if (!tip) return null;
  return <div className="ctip" style={{ left: tip.x, top: tip.y }}>{tip.body}</div>;
}

/* ------------------------------------------------------------------ chart 1: ranked horizontal bars (axis starts at zero) */
function RankBars({ rows, fmt, better }) {
  const [ref, w] = useWidth(); const [tip, setTip] = useState(null);
  const have = rows.filter((r) => r.value != null).sort((a, b) => (better === "low" ? a.value - b.value : b.value - a.value));
  const missing = rows.filter((r) => r.value == null);
  const L = Math.min(170, Math.max(104, w * 0.27)), R = 84, rowH = 30, T = 8, B = 30;
  const max = Math.max(...have.map((r) => r.value), 1e-9) * 1.04;
  const ts = ticksFor(0, max, 4); const top = ts[ts.length - 1];
  const x = (v) => L + ((w - L - R) * v) / top;
  const H = T + have.length * rowH + B;
  return (
    <div className="chartwrap" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Ranked bar chart">
        {ts.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={T} y2={H - B} stroke="var(--line)" strokeWidth="1" />
            <text x={x(t)} y={H - B + 16} textAnchor="middle" className="axt">{+t.toFixed(4)}</text>
          </g>
        ))}
        {have.map((r, i) => {
          const y = T + i * rowH;
          return (
            <g key={r.id} onMouseMove={(e) => setTip({ x: e.nativeEvent.offsetX + 12, y: y - 6, body: <><b>{r.name}</b><br />{fmt(r.value)}{r.note && <><br /><span className="muted2">{r.note}</span></>}</> })} onMouseLeave={() => setTip(null)}>
              <rect x={0} y={y} width={w} height={rowH} fill="transparent" />
              <text x={L - 10} y={y + rowH / 2 + 4} textAnchor="end" className="rowt">{r.name}</text>
              <path d={`M${x(0)},${y + 5} H${x(r.value) - 4} a4,4 0 0 1 4,4 v${rowH - 20} a4,4 0 0 1 -4,4 H${x(0)} Z`} fill={r.color} />
              <text x={x(r.value) + 8} y={y + rowH / 2 + 4} className="valt">{fmt(r.value)}</text>
            </g>
          );
        })}
      </svg>
      {missing.length > 0 && <p className="muted small">Not reported in the paper for this metric: {missing.map((m) => m.name).join(", ")}.</p>}
      <Tip tip={tip} />
    </div>
  );
}

/* ------------------------------------------------------------------ chart 2: one line per model across categories (data sources or disease classes) */
function LineChart({ cats, series, fmt, yLabel }) {
  const [ref, w] = useWidth(); const [tip, setTip] = useState(null); const [hov, setHov] = useState(null); const [hi, setHi] = useState(null);
  const H = 340, L = 56, R = 18, T = 14, B = 40;
  const vals = series.flatMap((s) => s.values).filter((v) => v != null);
  if (!vals.length) return <p className="muted">Select at least one model.</p>;
  const lo0 = Math.min(...vals), hi0 = Math.max(...vals), pad = (hi0 - lo0) * 0.12 || 0.02;
  const ts = ticksFor(lo0 - pad, hi0 + pad, 6); const lo = ts[0], hiT = ts[ts.length - 1];
  const dec = decimals(ts.length > 1 ? ts[1] - ts[0] : 0.05);
  const px = (i) => L + ((w - L - R) * (i + 0.5)) / cats.length, py = (v) => T + (H - T - B) * (1 - (v - lo) / (hiT - lo));
  return (
    <div className="chartwrap" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Line chart">
        {ts.map((t) => (
          <g key={t}><line x1={L} x2={w - R} y1={py(t)} y2={py(t)} stroke="var(--line)" /><text x={L - 8} y={py(t) + 4} textAnchor="end" className="axt">{t.toFixed(dec)}</text></g>
        ))}
        <text transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" className="axt">{yLabel}</text>
        {cats.map((c, i) => <text key={c} x={px(i)} y={H - 14} textAnchor="middle" className="axt">{c}</text>)}
        {hov != null && <line x1={px(hov)} x2={px(hov)} y1={T} y2={H - B} stroke="var(--muted)" strokeDasharray="3 3" />}
        {series.map((s) => {
          const dim = hi && hi !== s.id; const pts = s.values.map((v, i) => (v == null ? null : [px(i), py(v)]));
          return (
            <g key={s.id} opacity={dim ? 0.18 : 1}>
              <polyline fill="none" stroke={s.color} strokeWidth={s.id === hi ? 3 : 2} points={pts.filter(Boolean).map((p) => p.join(",")).join(" ")} />
              {pts.map((p, i) => p && <Mark key={i} kind={MODEL_MARK[s.id]} x={p[0]} y={p[1]} r={s.id === hi ? 5.6 : 4.6} color={s.color} />)}
            </g>
          );
        })}
        {cats.map((c, i) => (
          <rect key={c} x={px(i) - (w - L - R) / cats.length / 2} y={T} width={(w - L - R) / cats.length} height={H - T - B} fill="transparent"
            onMouseMove={(e) => {
              const rows = series.filter((s) => s.values[i] != null).sort((a, b) => b.values[i] - a.values[i]);
              setHov(i);
              setTip({ x: Math.min(px(i) + 14, w - 210), y: 20, body: <><b>{c}</b>{rows.map((s) => <div className="trow" key={s.id}><Swatch id={s.id} size={11} /><span>{s.name}</span><b>{fmt(s.values[i])}</b></div>)}</> });
            }}
            onMouseLeave={() => { setHov(null); setTip(null); }} />
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.id} className="lg" onMouseEnter={() => setHi(s.id)} onMouseLeave={() => setHi(null)}><Swatch id={s.id} /> {s.name}</span>
        ))}
      </div>
      <Tip tip={tip} />
    </div>
  );
}

/* ------------------------------------------------------------------ chart 3: trade-off scatter */
function Scatter({ pts, xLabel, yLabel, xFmt, yFmt, logx }) {
  const [ref, w] = useWidth(); const [tip, setTip] = useState(null);
  const H = 360, L = 58, R = 120, T = 16, B = 46;
  if (!pts.length) return <p className="muted">Select at least one model.</p>;
  const tx = (v) => (logx ? Math.log10(v) : v);
  const xs = pts.map((p) => tx(p.x)), ys = pts.map((p) => p.y);
  const xlo0 = Math.min(...xs), xhi0 = Math.max(...xs), ylo0 = Math.min(...ys), yhi0 = Math.max(...ys);
  const xp = (xhi0 - xlo0) * 0.12 || 0.5, yp = (yhi0 - ylo0) * 0.25 || 0.01;
  const xt = logx ? [1, 3, 10, 30, 100, 300].filter((t) => Math.log10(t) >= xlo0 - xp && Math.log10(t) <= xhi0 + xp) : ticksFor(xlo0 - xp, xhi0 + xp, 5);
  const yt = ticksFor(ylo0 - yp, yhi0 + yp, 5);
  const xlo = logx ? xlo0 - xp : xt[0], xhi = logx ? xhi0 + xp : xt[xt.length - 1];
  const ylo = yt[0], yhi = yt[yt.length - 1];
  const X = (v) => L + ((w - L - R) * (tx(v) - xlo)) / (xhi - xlo), Y = (v) => T + (H - T - B) * (1 - (v - ylo) / (yhi - ylo));
  const ydec = decimals(yt.length > 1 ? yt[1] - yt[0] : 0.01);
  return (
    <div className="chartwrap" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Scatter plot">
        {yt.map((t) => <g key={t}><line x1={L} x2={w - R} y1={Y(t)} y2={Y(t)} stroke="var(--line)" /><text x={L - 8} y={Y(t) + 4} textAnchor="end" className="axt">{t.toFixed(ydec)}</text></g>)}
        {xt.map((t) => { const xv = logx ? X(t) : X(t); return <text key={t} x={xv} y={H - B + 18} textAnchor="middle" className="axt">{t}</text>; })}
        <text x={(L + w - R) / 2} y={H - 6} textAnchor="middle" className="axt">{xLabel}</text>
        <text transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" className="axt">{yLabel}</text>
        {pts.map((p, i) => (
          <g key={p.id} onMouseMove={(e) => setTip({ x: Math.min(X(p.x) + 14, w - 190), y: Y(p.y) - 8, body: <><b>{p.name}</b><br />{xLabel}: {xFmt(p.x)}<br />{yLabel}: {yFmt(p.y)}</> })} onMouseLeave={() => setTip(null)}>
            <circle cx={X(p.x)} cy={Y(p.y)} r={14} fill="transparent" />
            <Mark kind={MODEL_MARK[p.id]} x={X(p.x)} y={Y(p.y)} r={6} color={p.color} />
            <text x={X(p.x) + (X(p.x) > w - R - 70 ? -11 : 11)} textAnchor={X(p.x) > w - R - 70 ? "end" : "start"} y={Y(p.y) + (i % 2 ? 16 : -9)} className="rowt small">{p.name}</text>
          </g>
        ))}
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

/* ------------------------------------------------------------------ chart 4: forest plot of paired bootstrap differences */
function Forest({ rows }) {
  const [ref, w] = useWidth(); const [tip, setTip] = useState(null);
  const L = Math.min(190, Math.max(120, w * 0.32)), R = 20, rowH = 34, T = 10, B = 44, H = T + rows.length * rowH + B;
  const lo = Math.min(...rows.map((r) => r.lo), 0) - 0.004, hi = Math.max(...rows.map((r) => r.hi), 0) + 0.004;
  const ts = ticksFor(lo, hi, 6); const a = ts[0], b = ts[ts.length - 1];
  const x = (v) => L + ((w - L - R) * (v - a)) / (b - a);
  const dec = decimals(ts.length > 1 ? ts[1] - ts[0] : 0.01);
  return (
    <div className="chartwrap" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Forest plot of paired differences">
        {ts.map((t) => <g key={t}><line x1={x(t)} x2={x(t)} y1={T} y2={H - B} stroke="var(--line)" /><text x={x(t)} y={H - B + 16} textAnchor="middle" className="axt">{t.toFixed(dec)}</text></g>)}
        <line x1={x(0)} x2={x(0)} y1={T} y2={H - B} stroke="var(--muted)" strokeWidth="1.5" />
        <text x={(L + w - R) / 2} y={H - 6} textAnchor="middle" className="axt">Difference in macro-AUROC (95% paired-bootstrap interval)</text>
        {rows.map((r, i) => {
          const y = T + i * rowH + rowH / 2, clear = r.lo > 0 || r.hi < 0, c = r.group === "A" ? "#2a78d6" : "#008300";
          return (
            <g key={r.label} onMouseMove={(e) => setTip({ x: Math.min(x(r.hi) + 10, w - 230), y: y - 10, body: <><b>{r.label}</b><br />{r.delta >= 0 ? "+" : ""}{r.delta.toFixed(4)} [{r.lo.toFixed(4)}, {r.hi.toFixed(4)}]<br />{clear ? "Interval excludes 0: clear difference" : "Interval includes 0: not separable"}</> })} onMouseLeave={() => setTip(null)}>
              <rect x={0} y={y - rowH / 2} width={w} height={rowH} fill="transparent" />
              <text x={L - 10} y={y + 4} textAnchor="end" className="rowt">{r.label}</text>
              <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} stroke={c} strokeWidth="2.4" strokeLinecap="round" />
              <circle cx={x(r.delta)} cy={y} r={5.5} fill={clear ? c : "var(--card)"} stroke={c} strokeWidth="2" />
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span className="lg"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#2a78d6" /></svg> CNN baselines (partition A)</span>
        <span className="lg"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#008300" /></svg> RETFound-based models (partition B)</span>
        <span className="lg"><svg width="14" height="14"><circle cx="7" cy="7" r="4.4" fill="var(--card)" stroke="#555" strokeWidth="2" /></svg> hollow = interval includes 0</span>
      </div>
      <Tip tip={tip} />
    </div>
  );
}

/* ------------------------------------------------------------------ page */
export default function Compare() {
  const [d, setD] = useState(null);
  const [view, setView] = useState("rank");
  const [metric, setMetric] = useState("auroc");
  const [scope, setScope] = useState("pooled");
  const [clsMetric, setClsMetric] = useState("auroc");
  const [srcMetric, setSrcMetric] = useState("auroc");
  const [xAxis, setXAxis] = useState("trainable");
  const [on, setOn] = useState(MODEL_ORDER);
  useEffect(() => { fetch("/api/model-info").then((r) => r.json()).then(setD).catch(() => {}); }, []);
  const models = useMemo(() => (d ? [...d.models].sort((a, b) => MODEL_ORDER.indexOf(a.id) - MODEL_ORDER.indexOf(b.id)) : []), [d]);
  if (!d) return <div className="page"><p className="muted">Loading...</p></div>;

  const shown = models.filter((m) => on.includes(m.id));
  const mdef = METRICS.find((m) => m.k === metric);
  const scopeNow = mdef.scoped ? scope : "pooled";
  const toggle = (id) => setOn((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const tag = (m) => m.name + " (" + m.partition + ")";

  const rankRows = shown.map((m) => { const v = mdef.get(m, scopeNow); return { id: m.id, name: tag(m), color: MODEL_COLOR[m.id], value: v, note: "Test partition " + m.partition }; });
  const lineBase = (key, cats, getter, fmt) => ({
    cats, fmt, series: shown.map((m) => ({ id: m.id, name: tag(m), color: MODEL_COLOR[m.id], values: getter(m, key) })),
  });
  const srcData = lineBase(srcMetric, ["Pooled", ...d.sources], (m, k) => [m.pooled[k], m.source.D1[k], m.source.D2[k], m.source.D3[k]], (v) => v.toFixed(4));
  const clsShown = shown.filter((m) => m.per_class);
  const clsData = { cats: d.classes, fmt: (v) => v.toFixed(4), series: clsShown.map((m) => ({ id: m.id, name: tag(m), color: MODEL_COLOR[m.id], values: m.per_class[clsMetric] })) };
  const scatterPts = shown.filter((m) => (xAxis === "latency" ? m.report : true)).map((m) => ({ id: m.id, name: m.name, color: MODEL_COLOR[m.id], x: xAxis === "latency" ? m.report.latency_ms : m.trainable_m, y: m.pooled.auroc }));

  return (
    <div className="page">
      <section className="card">
        <h2>Compare the models</h2>
        <p className="muted" style={{ marginTop: 0 }}>Every number on this page is copied from the paper (single training run per model). Pick which models to show; each model keeps the same colour and marker everywhere.</p>
        <div className="chips2">
          {models.map((m) => (
            <button key={m.id} className={"chip " + (on.includes(m.id) ? "on" : "")} onClick={() => toggle(m.id)} aria-pressed={on.includes(m.id)}>
              <Swatch id={m.id} /> {m.name} <small>{m.partition}</small>
            </button>
          ))}
          <button className="ghost" onClick={() => setOn(MODEL_ORDER)}>All</button>
          <button className="ghost" onClick={() => setOn(["2c", "concat", "attention", "retfound"])}>Transformer-based only</button>
        </div>
        <p className="muted small">A = CNN baselines and 2c, tested on the pooled split file. B = RETFound alone, Concat and Cross-attention, tested on the per-source split files. The two partitions have different test images, so comparing an A model with a B model is indicative only.</p>
      </section>

      <div className="tabs">
        {VIEWS.map(([k, label]) => <button key={k} className={view === k ? "on" : ""} onClick={() => setView(k)}>{label}</button>)}
      </div>

      {view === "rank" && (
        <section className="card">
          <div className="controls">
            <label>Metric
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                {["Ranking quality", "Operating point", "Cost"].map((g) => (
                  <optgroup key={g} label={g}>{METRICS.filter((m) => m.group === g).map((m) => <option key={m.k} value={m.k}>{m.label}</option>)}</optgroup>
                ))}
              </select>
            </label>
            {mdef.scoped && (
              <label>Test set
                <select value={scope} onChange={(e) => setScope(e.target.value)}>{SCOPES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
              </label>
            )}
          </div>
          <h3>{mdef.label}{mdef.scoped ? " on " + SCOPES.find((s) => s[0] === scopeNow)[1].toLowerCase() : ""} <small>({mdef.best === "low" ? "lower is better" : "higher is better"})</small></h3>
          <RankBars rows={rankRows} fmt={mdef.fmt} better={mdef.best} />
          <p className="muted small">Bars start at zero. Accuracy, precision, recall, specificity, latency and size were only measured for the four transformer-based models.</p>
          <DataTable cols={["Model", mdef.label]} rows={rankRows.map((r) => [r.name, r.value == null ? "not reported" : mdef.fmt(r.value)])} />
        </section>
      )}

      {view === "source" && (
        <section className="card">
          <div className="controls">
            <label>Metric
              <select value={srcMetric} onChange={(e) => setSrcMetric(e.target.value)}><option value="auroc">Macro-AUROC</option><option value="f1">Macro-F1</option></select>
            </label>
          </div>
          <h3>{srcMetric === "auroc" ? "Macro-AUROC" : "Macro-F1"} on each data source</h3>
          <LineChart cats={srcData.cats} series={srcData.series} fmt={srcData.fmt} yLabel={srcMetric === "auroc" ? "Macro-AUROC" : "Macro-F1"} />
          <p className="muted small">Every model finds D1 (ODIR-5K) hardest. The vertical axis is cut to show the differences; read the values from the tooltip or the table.</p>
          <DataTable cols={["Model", ...srcData.cats]} rows={srcData.series.map((s) => [s.name, ...s.values.map((v) => v.toFixed(4))])} />
        </section>
      )}

      {view === "class" && (
        <section className="card">
          <div className="controls">
            <label>Metric
              <select value={clsMetric} onChange={(e) => setClsMetric(e.target.value)}><option value="auroc">AUROC</option><option value="f1">F1</option></select>
            </label>
          </div>
          <h3>{clsMetric === "auroc" ? "AUROC" : "F1"} for each disease, pooled test set</h3>
          <LineChart cats={clsData.cats} series={clsData.series} fmt={clsData.fmt} yLabel={clsMetric === "auroc" ? "AUROC" : "F1"} />
          <p className="muted small">Per-class values were not exported for 2c, so it is not shown here. The vertical axis is cut.</p>
          <DataTable cols={["Model", ...clsData.cats]} rows={clsData.series.map((s) => [s.name, ...s.values.map((v) => v.toFixed(4))])} />
        </section>
      )}

      {view === "trade" && (
        <section className="card">
          <div className="controls">
            <label>Horizontal axis
              <select value={xAxis} onChange={(e) => setXAxis(e.target.value)}>
                <option value="trainable">Trainable parameters (M, log scale), all models</option>
                <option value="latency">Latency (ms / image), transformer-based models</option>
              </select>
            </label>
          </div>
          <h3>Accuracy against cost</h3>
          <Scatter pts={scatterPts} xLabel={xAxis === "latency" ? "Latency (ms / image)" : "Trainable parameters (M)"} yLabel="Macro-AUROC" xFmt={(v) => v.toFixed(xAxis === "latency" ? 2 : 1)} yFmt={(v) => v.toFixed(4)} logx={xAxis === "trainable"} />
          <p className="muted small">Up and to the left is better. 2c trains about a tenth of the parameters of the fully fine-tuned models but is the slowest at inference, because it runs two backbones and a larger CNN input.</p>
          <DataTable cols={["Model", xAxis === "latency" ? "Latency (ms)" : "Trainable (M)", "Macro-AUROC"]} rows={scatterPts.map((p) => [p.name, p.x.toFixed(2), p.y.toFixed(4)])} />
        </section>
      )}

      {view === "paired" && (
        <section className="card">
          <h3>Paired bootstrap differences in macro-AUROC <small>(1000 resamples, pooled test set)</small></h3>
          <Forest rows={d.paired} />
          <p className="muted small">Models are only compared inside their own partition, because both models must be tested on the same images. Adding a CNN branch to RETFound gives a clear gain (+0.012 to +0.014); Cross-attention versus Concat is not separable overall.</p>
          <DataTable cols={["Comparison", "Difference", "95% interval", "Clear?"]} rows={d.paired.map((r) => [r.label, r.delta.toFixed(4), `[${r.lo.toFixed(4)}, ${r.hi.toFixed(4)}]`, r.lo > 0 || r.hi < 0 ? "yes" : "no"])} />
        </section>
      )}

      <section className="card">
        <h2>Cautions</h2>
        <ul>{d.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      </section>
    </div>
  );
}

function DataTable({ cols, rows }) {
  return (
    <details className="dt">
      <summary>Show the numbers as a table</summary>
      <div className="scroll"><table><thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody></table></div>
    </details>
  );
}
