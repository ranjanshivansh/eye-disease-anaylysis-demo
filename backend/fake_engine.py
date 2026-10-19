"""Stand-in used only for UI testing when FAKE_MODEL=1 (no torch, no checkpoints). NOT the real models."""
import hashlib, time
import numpy as np
from PIL import Image
CLASSES = ["Normal", "DR", "Glaucoma", "AMD", "Myopia"]
_M = [("vgg16", "VGG16", "CNN baseline", False, "ImageNet VGG16, fully fine-tuned."),
      ("resnet50", "ResNet50", "CNN baseline", False, "ImageNet ResNet50, fully fine-tuned."),
      ("efficientnet_b3", "EfficientNet-B3", "CNN baseline", False, "ImageNet EfficientNet-B3, fully fine-tuned."),
      ("retfound", "RETFound alone", "Foundation model", False, "Retinal foundation model (ViT-L/16), fully fine-tuned."),
      ("concat", "Concat fusion", "Fusion", False, "RETFound + ResNet50, features concatenated."),
      ("attention", "Cross-attention fusion", "Fusion", False, "RETFound + ResNet50, bidirectional cross-attention with a modality gate."),
      ("2c", "2c (LoRA hybrid)", "Ours", True, "RETFound with LoRA + ResNet50 + cross-attention + 5 disease queries. Gives per-disease Grad-CAM.")]
BY_ID = {m[0]: m for m in _M}
IMG = 320


def catalogue():
    return [{"id": i, "name": n, "family": f, "description": d, "available": True, "missing": None, "heatmaps": h,
             "input_size": IMG, "threshold_source": "fake"} for i, n, f, h, d in _M]


def run(mid, img):
    img = img.resize((IMG, IMG)); arr = np.asarray(img).astype(np.float32)
    seed = int(hashlib.md5((str(arr.mean()) + mid).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed); time.sleep(0.4)
    probs = rng.uniform(0.05, 0.95, 5); thr = np.array([0.5, 0.4, 0.5, 0.5, 0.45])
    outs = None
    if BY_ID[mid][3]:
        yy, xx = np.mgrid[0:IMG, 0:IMG]; outs = []
        for _ in range(5):
            cy, cx = rng.uniform(80, 240, 2)
            h = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 40 ** 2)))
            heat = np.stack([h * 255, h * 80, (1 - h) * 160], -1)
            outs.append((0.55 * arr + 0.45 * heat).clip(0, 255).astype(np.uint8))
    return {"probs": probs, "thresholds": thr, "overlays": outs, "original": arr.astype(np.uint8), "seconds": 0.4}
