import io, os, json, time, base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.getenv("FAKE_MODEL") == "1"
if FAKE:
    import fake_engine as eng
    print("!! FAKE_MODEL=1: UI test mode, predictions are NOT from the real models", flush=True)
else:
    import engine as eng
import db

CLASSES = eng.CLASSES
STATIC = os.path.join(HERE, "static")
print("Ready. Open http://127.0.0.1:8000", flush=True)


def jpeg_b64(arr, size=None, q=88):
    im = Image.fromarray(arr)
    if size:
        im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


app = FastAPI(title="Retinal disease demo (research prototype)")


@app.get("/api/models")
def models():
    """Which models exist and which of them can run right now (checkpoint found)."""
    return {"fake": FAKE, "classes": CLASSES, "models": eng.catalogue()}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), models: str = Form("2c")):
    ids = [m.strip() for m in models.split(",") if m.strip()]
    cat = {m["id"]: m for m in eng.catalogue()}
    if not ids:
        raise HTTPException(400, "Select at least one model.")
    bad = [i for i in ids if i not in cat]
    if bad:
        raise HTTPException(400, "Unknown model: " + ", ".join(bad))
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read this file as an image.")
    t_all = time.time()
    out, original, store = [], None, []
    for mid in ids:
        info = cat[mid]
        if not info["available"]:
            out.append({"id": mid, "name": info["name"], "ok": False, "error": f"Checkpoint not found: {info['missing']}"}); continue
        try:
            r = eng.run(mid, img)
        except Exception as e:  # one broken model must not stop the others
            out.append({"id": mid, "name": info["name"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}); continue
        original = r["original"] if original is None else original
        probs, thr = r["probs"], r["thresholds"]
        flags = [bool(probs[j] >= thr[j]) for j in range(5)]
        out.append({"id": mid, "name": info["name"], "ok": True, "seconds": r["seconds"], "heatmaps": r["overlays"] is not None,
                    "threshold_source": info["threshold_source"],
                    "results": [{"name": CLASSES[j], "prob": float(probs[j]), "threshold": float(thr[j]), "positive": flags[j],
                                 "heatmap": jpeg_b64(r["overlays"][j]) if r["overlays"] else None} for j in range(5)]})
        store.append({"id": mid, "name": info["name"], "probs": [float(p) for p in probs], "flags": flags,
                      "thresholds": [float(t) for t in thr], "seconds": r["seconds"]})
    if not store:
        raise HTTPException(500, "; ".join(o.get("error", "") for o in out) or "No model could run.")
    first = store[0]
    with db.Session() as s:
        row = db.Scan(filename=(file.filename or "image")[:255], thumb=jpeg_b64(original, 160, 80),
                      probs=json.dumps(first["probs"]), flags=json.dumps(first["flags"]),
                      seconds=round(time.time() - t_all, 1), extra=json.dumps(store))
        s.add(row); s.commit(); sid = row.id
    return {"id": sid, "seconds": round(time.time() - t_all, 1), "original": jpeg_b64(original), "fake": FAKE, "models": out}


def _row(r):
    models = json.loads(r.extra) if r.extra else [{"id": "2c", "name": "2c (LoRA hybrid)", "probs": json.loads(r.probs),
                                                    "flags": json.loads(r.flags), "thresholds": [0.5] * 5, "seconds": r.seconds}]
    return {"id": r.id, "created_at": r.created_at.isoformat() + "Z", "filename": r.filename, "thumb": r.thumb,
            "seconds": r.seconds, "classes": CLASSES, "models": models}


@app.get("/api/history")
def history(limit: int = 100):
    with db.Session() as s:
        rows = s.query(db.Scan).order_by(db.Scan.id.desc()).limit(min(limit, 500)).all()
        return [_row(r) for r in rows]


@app.get("/api/history/{sid}")
def history_one(sid: int):
    with db.Session() as s:
        r = s.get(db.Scan, sid)
        if not r:
            raise HTTPException(404, "Not found")
        return _row(r)


@app.delete("/api/history/{sid}")
def history_delete(sid: int):
    with db.Session() as s:
        r = s.get(db.Scan, sid)
        if r:
            s.delete(r); s.commit()
    return {"ok": True}


@app.get("/api/model-info")
def model_info():
    with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
        info = json.load(f)
    info["database"] = db.DB_NAME
    info["fake"] = FAKE
    return info


if os.path.isdir(os.path.join(STATIC, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets")), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    f = os.path.join(STATIC, "index.html")
    if not os.path.exists(f):
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend folder."}
    return FileResponse(f)
