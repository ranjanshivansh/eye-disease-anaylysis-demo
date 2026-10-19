export const GUIDE = {
  Normal: "No single focal spot is expected. A diffuse or scattered map is typical.",
  DR: "Lesions spread across the retina (haemorrhages, exudates). Look for several bright patches, not one.",
  Glaucoma: "Optic disc and cup region (the bright round area where vessels converge).",
  AMD: "Macula (the darker central area, to the side of the disc).",
  Myopia: "Area around the optic disc and the surrounding background (tilted disc, tessellated fundus).",
};

export const pct = (x) => (x * 100).toFixed(1) + "%";

export function Bars({ classes, probs, thresholds, flags }) {
  return (
    <div className="bars">
      {classes.map((c, i) => (
        <div className="barrow" key={c}>
          <div className="bname">{c}</div>
          <div className="track">
            <div className={"fill " + (flags[i] ? "hot" : "")} style={{ width: probs[i] * 100 + "%" }} />
            <div className="tick" style={{ left: thresholds[i] * 100 + "%" }} title={"decision threshold " + thresholds[i].toFixed(2)} />
          </div>
          <div className="bval">{pct(probs[i])}</div>
          <div className={"badge " + (flags[i] ? "yes" : "no")}>{flags[i] ? "Flagged" : "Not flagged"}</div>
        </div>
      ))}
      <div className="hint">The vertical tick on each bar is the decision threshold tuned on the validation set. A disease is flagged when its probability reaches the tick.</div>
    </div>
  );
}

// fixed colour + marker per model (never re-assigned when models are hidden)
export const MODEL_ORDER = ["vgg16", "resnet50", "efficientnet_b3", "retfound", "concat", "attention", "2c"];
export const MODEL_COLOR = {
  vgg16: "#2a78d6", resnet50: "#eb6834", efficientnet_b3: "#1baf7a", retfound: "#eda100",
  concat: "#e87ba4", attention: "#008300", "2c": "#6250d6",
};
export const MODEL_MARK = { vgg16: "circle", resnet50: "square", efficientnet_b3: "triangle", retfound: "diamond", concat: "down", attention: "plus", "2c": "cross" };

export function Mark({ kind, x, y, r = 4.6, color, ring = true }) {
  const stroke = ring ? "var(--card)" : "none";
  const p = { fill: color, stroke, strokeWidth: 1.4 };
  switch (kind) {
    case "square": return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} {...p} />;
    case "triangle": return <polygon points={`${x},${y - r * 1.15} ${x + r * 1.1},${y + r * 0.85} ${x - r * 1.1},${y + r * 0.85}`} {...p} />;
    case "down": return <polygon points={`${x},${y + r * 1.15} ${x + r * 1.1},${y - r * 0.85} ${x - r * 1.1},${y - r * 0.85}`} {...p} />;
    case "diamond": return <polygon points={`${x},${y - r * 1.25} ${x + r * 1.25},${y} ${x},${y + r * 1.25} ${x - r * 1.25},${y}`} {...p} />;
    case "plus": return <path d={`M${x - r},${y} H${x + r} M${x},${y - r} V${y + r}`} stroke={color} strokeWidth={2.6} fill="none" />;
    case "cross": return <path d={`M${x - r},${y - r} L${x + r},${y + r} M${x - r},${y + r} L${x + r},${y - r}`} stroke={color} strokeWidth={2.6} fill="none" />;
    default: return <circle cx={x} cy={y} r={r} {...p} />;
  }
}

export function Swatch({ id, size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true">
      <Mark kind={MODEL_MARK[id]} x={7} y={7} r={4} color={MODEL_COLOR[id]} ring={false} />
    </svg>
  );
}
