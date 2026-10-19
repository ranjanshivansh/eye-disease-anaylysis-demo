import { useEffect, useState } from "react";
import { Swatch } from "../shared.jsx";
const f = (x) => (x == null ? "-" : x.toFixed(4));

export default function ModelPage() {
  const [d, setD] = useState(null);
  useEffect(() => { fetch("/api/model-info").then((r) => r.json()).then(setD).catch(() => {}); }, []);
  if (!d) return <div className="page"><p className="muted">Loading...</p></div>;
  return (
    <div className="page">
      <section className="card">
        <h2>How the model works</h2>
        <div className="flow">
          <div><b>RETFound ViT-L</b><small>retinal foundation model, adapted with LoRA (LoRA, small trainable share)</small></div>
          <span>+</span>
          <div><b>ResNet50</b><small>local texture features at 320 px</small></div>
          <span>&rarr;</span>
          <div><b>Cross-attention fusion</b><small>ViT tokens query the CNN feature map</small></div>
          <span>&rarr;</span>
          <div><b>5 disease queries</b><small>one learned query per disease, sigmoid output</small></div>
        </div>
        <p className="muted">Trainable parameters: {d.trainable.ours} compared with {d.trainable.full_finetune} for full fine-tuning ({d.trainable.ratio}). Trained on D123 (ODIR5K + MuReD + RFMiD), multi-label, focal loss, thresholds tuned on validation.</p>
      </section>

      <section className="card">
        <h2>Test-set results, pooled</h2>
        <div className="scroll"><table>
          <thead><tr><th>Model</th><th>Test partition</th><th>Macro-AUROC</th><th>AP</th><th>Macro-F1</th></tr></thead>
          <tbody>{d.models.map((m) => (
            <tr key={m.id} className={m.id === "2c" ? "ours" : ""}><td><Swatch id={m.id} /> {m.name}</td><td>{m.partition}</td><td>{f(m.pooled.auroc)}</td><td>{f(m.pooled.ap)}</td><td>{f(m.pooled.f1)}</td></tr>
          ))}</tbody>
        </table></div>
      </section>

      <section className="card">
        <h2>2c, by data source</h2>
        <div className="scroll"><table>
          <thead><tr><th>Source</th><th>Macro-AUROC</th><th>Macro-F1</th></tr></thead>
          <tbody>{d.sources.map((s, i) => { const x = d.models.find((q) => q.id === "2c").source["D" + (i + 1)]; return (
            <tr key={s}><td>{s}</td><td>{f(x.auroc)}</td><td>{f(x.f1)}</td></tr>
          ); })}</tbody>
        </table></div>
        <p className="muted">See the Compare models tab for every model, per source and per disease.</p>
      </section>

      <section className="card">
        <h2>Limitations</h2>
        <ul>{d.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        <p className="muted">History storage: {d.database}.</p>
      </section>
    </div>
  );
}
