"""Network architectures of all models in the study (copied from the training notebooks; no pretrained downloads:
the weights always come from your own checkpoint files)."""
import torch, torch.nn as nn, torch.nn.functional as F, timm
from torchvision import models as tvm

try:  # only matters if an incompatible torchao is installed
    import peft.tuners.lora.torchao as _pt; _pt.is_torchao_available = lambda: False
except Exception:
    pass
from peft import LoraConfig, get_peft_model

NUM_CLASSES = 5


# ----------------------------------------------------------------------------- CNN baselines (torchvision, last layer replaced)
def build_cnn(kind):
    if kind == "vgg16":
        m = tvm.vgg16(weights=None); m.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    elif kind == "resnet50":
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif kind == "efficientnet_b3":
        m = tvm.efficientnet_b3(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(kind)
    return m


# ----------------------------------------------------------------------------- RETFound alone (ViT-L/16, full fine-tune)
def build_retfound(img=224):
    return timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=NUM_CLASSES, img_size=img, global_pool="avg")


# ----------------------------------------------------------------------------- RETFound + ResNet50, concatenation fusion
class ConcatFusion(nn.Module):
    def __init__(self, num_classes=5, img=224, dim=512):
        super().__init__()
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, img_size=img, global_pool="avg")
        self.cnn = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=0)
        vd, cd = self.vit.num_features, self.cnn.num_features
        self.vit_proj = nn.Sequential(nn.LayerNorm(vd), nn.Linear(vd, dim), nn.GELU())
        self.cnn_proj = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * dim, num_classes))

    def forward(self, x):
        return self.classifier(torch.cat([self.vit_proj(self.vit(x)), self.cnn_proj(self.cnn(x))], 1))


# ----------------------------------------------------------------------------- RETFound + ResNet50, bidirectional cross-attention + gate
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, mlp_ratio=4):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ffn = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, q, ctx):
        c = self.norm_kv(ctx)
        a, _ = self.attn(self.norm_q(q), c, c, need_weights=False)
        x = q + a
        return x + self.ffn(self.norm_ffn(x))


class AttnFusion(nn.Module):
    def __init__(self, num_classes=5, img=224, dim=512, heads=8, layers=1, adrop=0.1):
        super().__init__()
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, img_size=img, global_pool="avg")
        self.cnn = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=0)
        vd, cd = self.vit.num_features, self.cnn.num_features
        self.vit_token_proj = nn.Sequential(nn.LayerNorm(vd), nn.Linear(vd, dim))
        self.cnn_token_proj = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, dim))
        self.cnn_pos_embed = nn.Parameter(torch.zeros(1, (img // 32) ** 2, dim))
        self.vit_attends_cnn = nn.ModuleList([CrossAttentionBlock(dim, heads, adrop) for _ in range(layers)])
        self.cnn_attends_vit = nn.ModuleList([CrossAttentionBlock(dim, heads, adrop) for _ in range(layers)])
        self.modality_gate = nn.Linear(2 * dim, 2)
        self.fused_norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def forward(self, x):
        vt = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]
        ct = self.cnn.forward_features(x).flatten(2).transpose(1, 2)
        vt = self.vit_token_proj(vt.float()); ct = self.cnn_token_proj(ct.float()) + self.cnn_pos_embed
        for vb, cb in zip(self.vit_attends_cnn, self.cnn_attends_vit):
            vt, ct = vb(vt, ct), cb(ct, vt)
        vv, cv = vt.mean(1), ct.mean(1)
        g = torch.softmax(self.modality_gate(torch.cat([vv, cv], 1)), 1)
        return self.classifier(self.fused_norm(g[:, :1] * vv + g[:, 1:] * cv))


# ----------------------------------------------------------------------------- 2c: RETFound(LoRA) + ResNet50 + cross-attention + disease queries
def _ffn(D):
    return nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * D, D))


class Hybrid(nn.Module):
    def __init__(s, vit, cnn, D=512, ncls=5, n_tok=100):
        super().__init__()
        s.vit, s.cnn = vit, cnn
        dv, dc = 1024, cnn.num_features
        s.pv, s.pc = nn.Linear(dv, D), nn.Linear(dc, D)
        s.pos_c = nn.Parameter(torch.randn(1, n_tok, D) * .02)
        s.ln_q, s.ln_kv, s.ln_f = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        s.xattn = nn.MultiheadAttention(D, 8, dropout=0.1, batch_first=True)
        s.ffn = _ffn(D)
        s.q = nn.Parameter(torch.randn(ncls, D) * .02)
        s.ln_t, s.ln_d, s.ln_d2 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        s.dec = nn.MultiheadAttention(D, 8, dropout=0.1, batch_first=True)
        s.ffn_d = _ffn(D)
        s.cls_w = nn.Parameter(torch.randn(ncls, D) * .02)
        s.cls_b = nn.Parameter(torch.zeros(ncls))


def build_2c():
    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="avg", img_size=224)
    vit = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["qkv"]))
    return Hybrid(vit, timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool=""))


def gradcam_2c(s, x, vit_img=224):
    """One pass: probabilities + class-specific Grad-CAM (14x14) for each disease (2c only)."""
    xv = F.interpolate(x, size=vit_img, mode="bilinear", align_corners=False)
    with torch.no_grad():
        tv = s.vit.base_model.model.forward_features(xv)
        fm = s.cnn.forward_features(x)
    tv = tv.clone().requires_grad_(True)
    with torch.enable_grad():
        L = s.ln_kv(s.pc(fm.flatten(2).transpose(1, 2)) + s.pos_c)
        Tk = s.pv(tv)
        a, _ = s.xattn(s.ln_q(Tk), L, L); Tk = Tk + a
        Tk = Tk + s.ffn(s.ln_f(Tk))
        Q = s.q.unsqueeze(0).expand(x.size(0), -1, -1); kv = s.ln_t(Tk)
        d, _ = s.dec(s.ln_d(Q), kv, kv, need_weights=False)
        Q = Q + d; Q = Q + s.ffn_d(s.ln_d2(Q))
        logits = (Q * s.cls_w).sum(-1) + s.cls_b
        cams = []
        for c in range(5):
            g, = torch.autograd.grad(logits[0, c], tv, retain_graph=True)
            alpha = g[0, 1:].mean(0)
            cams.append(torch.relu((tv[0, 1:] * alpha).sum(-1)).reshape(14, 14))
    return torch.sigmoid(logits.detach())[0], torch.stack(cams).detach()
