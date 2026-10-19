"""Model registry + inference for every model of the study.

A model is 'available' when its checkpoint file is found (in backend/checkpoints/ or in backend/).
Models are loaded on first use and kept in a small cache (MAX_LOADED, default 2) because the ViT-L models are large.
Only the 2c model produces class-specific Grad-CAM heatmaps; the other models return probabilities only.
"""
import os, json, time, threading
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from matplotlib import colormaps
import models as M

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIRS = [os.getenv("EYE_CKPT_DIR", ""), os.path.join(HERE, "checkpoints"), HERE]
CLASSES = ["Normal", "DR", "Glaucoma", "AMD", "Myopia"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("USE_GPU", "1") != "0" else "cpu")
MAX_LOADED = max(1, int(os.getenv("MAX_LOADED", "2")))
CNN_SIZE = int(os.getenv("CNN_SIZE", "300"))   # input size the CNN baselines were trained at: check against your baseline notebook (IMG_SIZE)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
torch.set_num_threads(max(1, os.cpu_count() or 1))

# validation-tuned thresholds printed in the training notebooks (used when no thresholds file is next to the checkpoint)
THR_FALLBACK = {"retfound": [0.67, 0.42, 0.67, 0.91, 0.54],
                "concat": [0.21, 0.60, 0.64, 0.40, 0.25],
                "attention": [0.52, 0.41, 0.71, 0.89, 0.89]}

# id, display name, family, checkpoint file, input size, builder
REGISTRY = [
    dict(id="vgg16", name="VGG16", family="CNN baseline", ckpt="vgg16_full_best.pth", size=CNN_SIZE, build=lambda: M.build_cnn("vgg16"),
         desc="ImageNet VGG16, fully fine-tuned."),
    dict(id="resnet50", name="ResNet50", family="CNN baseline", ckpt="resnet50_full_best.pth", size=CNN_SIZE, build=lambda: M.build_cnn("resnet50"),
         desc="ImageNet ResNet50, fully fine-tuned."),
    dict(id="efficientnet_b3", name="EfficientNet-B3", family="CNN baseline", ckpt="efficientnet_b3_full_best.pth", size=CNN_SIZE,
         build=lambda: M.build_cnn("efficientnet_b3"), desc="ImageNet EfficientNet-B3, fully fine-tuned."),
    dict(id="retfound", name="RETFound alone", family="Foundation model", ckpt="retfound_D123_best.pth", size=224, build=M.build_retfound,
         desc="Retinal foundation model (ViT-L/16), fully fine-tuned."),
    dict(id="concat", name="Concat fusion", family="Fusion", ckpt="fusion_D123_best.pth", size=224, build=lambda: M.ConcatFusion(),
         desc="RETFound + ResNet50, features concatenated."),
    dict(id="attention", name="Cross-attention fusion", family="Fusion", ckpt="attn_fusion_D123_best.pth", size=224, build=lambda: M.AttnFusion(),
         desc="RETFound + ResNet50, bidirectional cross-attention with a modality gate."),
    dict(id="2c", name="2c (LoRA hybrid)", family="Ours", ckpt="hybrid_xattn_query_best.pth", size=320, build=M.build_2c, heatmaps=True,
         desc="RETFound with LoRA + ResNet50 + cross-attention + 5 disease queries. Gives per-disease Grad-CAM."),
]
BY_ID = {m["id"]: m for m in REGISTRY}


def _find(name):
    for d in CKPT_DIRS:
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)


def _thr_file(m):
    stem = m["ckpt"].rsplit(".", 1)[0]
    for name in (stem.replace("_best", "") + "_thresholds.npy", stem.replace("_best", "_thresholds") + ".npy",
                 m["id"] + "_thresholds.json", stem + "_thresholds.json", "summary_" + m["id"] + ".json"):
        p = _find(name)
        if p:
            return p


def thresholds_for(m):
    """returns (array of 5, source text)"""
    p = _thr_file(m)
    if p and p.endswith(".npy"):
        return np.load(p).astype(float), "file " + os.path.basename(p)
    if p and p.endswith(".json"):
        d = json.load(open(p)); d = d.get("thresholds", d)
        return np.array([float(d[c]) for c in CLASSES]), "file " + os.path.basename(p)
    if m["id"] in THR_FALLBACK:
        return np.array(THR_FALLBACK[m["id"]], float), "validation-tuned values from the training notebook"
    return np.full(5, 0.5), "default 0.5 (no tuned thresholds file found)"


def catalogue():
    out = []
    for m in REGISTRY:
        path = _find(m["ckpt"])
        thr, src = thresholds_for(m)
        out.append({"id": m["id"], "name": m["name"], "family": m["family"], "description": m["desc"], "available": bool(path),
                    "missing": None if path else m["ckpt"], "heatmaps": bool(m.get("heatmaps")), "input_size": m["size"],
                    "threshold_source": src})
    return out


# ------------------------------------------------------------------------------------------------ loading
_cache, _order, _lock = {}, [], threading.Lock()


def _load(mid):
    with _lock:
        if mid in _cache:
            _order.remove(mid); _order.append(mid)
            return _cache[mid]
        m = BY_ID[mid]; path = _find(m["ckpt"])
        if not path:
            raise FileNotFoundError(f"{m['ckpt']} not found (put it in backend/checkpoints/)")
        print(f"Loading {m['name']} ...", flush=True)
        net = m["build"]()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd.get("model_state_dict", sd)) if isinstance(sd, dict) else sd
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        print(net.load_state_dict(sd, strict=True), flush=True)
        net.eval().to(DEVICE)
        thr, _ = thresholds_for(m)
        tf = T.Compose([T.Resize((m["size"], m["size"])), T.ToTensor(), T.Normalize(MEAN, STD)])
        _cache[mid] = (net, thr, tf, m["size"]); _order.append(mid)
        while len(_order) > MAX_LOADED:
            old = _order.pop(0); _cache.pop(old, None)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        return _cache[mid]


def _overlay(arr, cam, size):
    h = F.interpolate(cam[None, None].float().cpu(), size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    heat = colormaps["jet"](h)[..., :3] * 255
    return (0.55 * arr + 0.45 * heat).clip(0, 255).astype(np.uint8)


def run(mid, img: Image.Image):
    """img: RGB PIL image. Returns dict(probs[5], thresholds[5], overlays list|None, original uint8 array, seconds)."""
    net, thr, tf, size = _load(mid)
    shown = img.resize((320, 320)); arr = np.asarray(shown).astype(np.float32)
    t0 = time.time()
    x = tf(img).unsqueeze(0).to(DEVICE)
    overlays = None
    if BY_ID[mid].get("heatmaps"):
        probs, cams = M.gradcam_2c(net, x)
        probs = probs.cpu().numpy()
        overlays = [_overlay(arr, cams[j], 320) for j in range(5)]
    else:
        with torch.no_grad():
            probs = torch.sigmoid(net(x).float())[0].cpu().numpy()
    return {"probs": probs, "thresholds": thr, "overlays": overlays, "original": arr.astype(np.uint8), "seconds": round(time.time() - t0, 1)}
