"""One-page PDF report built with matplotlib (already a requirement), so no extra package and no internet are needed."""
import io, base64, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

INK, MUTED, LINE, HOT, ACC = "#0e1a24", "#566573", "#d8e0e6", "#b4381f", "#0a6c74"


def _img(data_url):
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        return None


def summary_lines(models, classes):
    """Plain sentences from the numbers only (no diagnosis wording)."""
    n = len(models)
    out = []
    flagged = [(c, [m["name"] for m in models if m["results"][j]["positive"]]) for j, c in enumerate(classes)]
    anyflag = [f for f in flagged if f[1]]
    if not anyflag:
        out.append("No disease reached its decision threshold in any selected model.")
    for c, who in anyflag:
        if n == 1:
            j = classes.index(c); p = models[0]["results"][j]["prob"]
            out.append(f"{who[0]} flags {c} ({p*100:.1f}%).")
        elif len(who) == n:
            out.append(f"All {n} models flag {c}.")
        else:
            out.append(f"{c} is flagged by {len(who)} of {n} models ({', '.join(who)}); the models disagree.")
    return out


def build(payload):
    models = [m for m in payload.get("models", []) if m.get("ok")]
    if not models:
        raise ValueError("No model results to report.")
    classes = [r["name"] for r in models[0]["results"]]
    original = _img(payload.get("original", "")) if payload.get("include_image", True) else None
    fn = str(payload.get("filename") or "image")[:80]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.955, "Retinal screening report", fontsize=20, fontweight="bold", color=INK)
        fig.text(0.07, 0.93, "Research prototype. Not a medical device. Not for diagnosis.", fontsize=10, color=HOT)
        fig.text(0.07, 0.91, f"File: {fn}    Created: {datetime.datetime.now():%d %b %Y %H:%M}", fontsize=9, color=MUTED)
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.90, 0.90], color=LINE, lw=1))
        top = 0.88
        if original is not None:
            ax = fig.add_axes([0.07, top - 0.25, 0.30, 0.25]); ax.imshow(original); ax.axis("off")
            ax.set_title("Uploaded image", fontsize=9, color=MUTED, loc="left")
            tx = 0.41
        else:
            tx = 0.07
        fig.text(tx, top - 0.01, "Summary", fontsize=12, fontweight="bold", color=INK, va="top")
        y = top - 0.04
        for line in summary_lines(models, classes):
            fig.text(tx, y, "- " + line, fontsize=9.5, color=INK, va="top", wrap=True); y -= 0.028
        fig.text(tx, y - 0.01, "Flags use each model's own threshold tuned on a validation set.", fontsize=8.5, color=MUTED, va="top")
        # table: diseases x models
        ty = top - 0.29
        fig.text(0.07, ty, "Probability per model (flagged = at or above its threshold)", fontsize=12, fontweight="bold", color=INK, va="top")
        ax = fig.add_axes([0.07, ty - 0.03 - 0.045 * (len(classes) + 1), 0.86, 0.045 * (len(classes) + 1)]); ax.axis("off")
        cols = ["Disease"] + [m["name"] for m in models]
        cell = [[c] + [("%.1f%%" % (m["results"][j]["prob"] * 100)) + ("  FLAGGED" if m["results"][j]["positive"] else "") for m in models] for j, c in enumerate(classes)]
        t = ax.table(cellText=cell, colLabels=cols, loc="upper left", cellLoc="left", colLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8 if len(models) > 4 else 9); t.scale(1, 1.6)
        for (r, c), cl in t.get_celld().items():
            cl.set_edgecolor(LINE)
            if r == 0:
                cl.set_text_props(fontweight="bold", color=MUTED); cl.set_facecolor("#f1f4f6")
            elif c > 0 and "FLAGGED" in cl.get_text().get_text():
                cl.set_text_props(color=HOT, fontweight="bold"); cl.set_facecolor("#f9e4de")
        # heatmaps (2c only)
        hm = next((m for m in models if m.get("heatmaps")), None)
        if hm is not None:
            hy = 0.30
            fig.text(0.07, hy + 0.03, f"Where {hm['name']} looked (Grad-CAM, qualitative)", fontsize=12, fontweight="bold", color=INK)
            for j, r in enumerate(hm["results"]):
                im = _img(r.get("heatmap") or "")
                if im is None:
                    continue
                ax = fig.add_axes([0.07 + j * 0.175, hy - 0.125, 0.16, 0.125]); ax.imshow(im); ax.axis("off")
                ax.set_title(r["name"] + (" (flagged)" if r["positive"] else ""), fontsize=8, color=HOT if r["positive"] else MUTED)
        fig.text(0.07, 0.085, "How to read this report", fontsize=10, fontweight="bold", color=INK)
        fig.text(0.07, 0.045,
                 "Probabilities are model outputs, not diagnoses. Models can disagree on the same image. Heatmaps show where a model looked and\n"
                 "do not prove that a finding is present. A qualified clinician must review the image. Single training run per model; no external test set.",
                 fontsize=8, color=MUTED, va="bottom")
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()
