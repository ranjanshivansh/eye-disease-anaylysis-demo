import { useEffect, useState } from "react";
import { Bars, Swatch } from "../shared.jsx";

export default function History() {
  const [rows, setRows] = useState(null);
  const [open, setOpen] = useState(null);
  const [err, setErr] = useState("");

  async function load() {
    try {
      const r = await fetch("/api/history");
      if (!r.ok) throw new Error("Server error " + r.status);
      setRows(await r.json());
    } catch (e) { setErr(e.message); }
  }
  useEffect(() => { load(); }, []);

  async function del(id) {
    await fetch("/api/history/" + id, { method: "DELETE" });
    setOpen(null); load();
  }

  if (err) return <div className="page"><div className="note err">{err}</div></div>;
  if (!rows) return <div className="page"><p className="muted">Loading...</p></div>;
  if (!rows.length) return <div className="page"><div className="card"><h2>No analyses yet</h2><p className="muted">Analyse an image on the Analyze tab and it will appear here. Only the probabilities, flags and a small thumbnail are stored, not the full image.</p></div></div>;

  return (
    <div className="page">
      <p className="muted">{rows.length} saved analyses. Only probabilities, flags and a small thumbnail are stored.</p>
      <div className="hist">
        {rows.map((r) => (
          <div className="card hrow" key={r.id}>
            <div className="hhead" onClick={() => setOpen(open === r.id ? null : r.id)}>
              <img src={r.thumb} alt="" />
              <div className="hmeta">
                <b>{r.filename}</b>
                <small>{new Date(r.created_at).toLocaleString()}</small>
                <div className="chips">
                  {r.models.map((m) => (
                    <span className="badge no" key={m.id}><Swatch id={m.id} size={11} /> {m.name}: {m.flags.some(Boolean) ? r.classes.filter((c, i) => m.flags[i]).join(", ") : "nothing flagged"}</span>
                  ))}
                </div>
              </div>
              <button className="ghost danger" onClick={(e) => { e.stopPropagation(); del(r.id); }}>Delete</button>
            </div>
            {open === r.id && r.models.map((m) => (
              <div key={m.id} style={{ marginTop: 16 }}>
                <b><Swatch id={m.id} /> {m.name}</b> <small className="muted">{m.seconds != null ? m.seconds + " s" : ""}</small>
                <Bars classes={r.classes} probs={m.probs} thresholds={m.thresholds} flags={m.flags} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
