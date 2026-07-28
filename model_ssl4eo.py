#!/usr/bin/env python3
"""
model_ssl4eo.py
===============
Multi-sensor SSL model (CROMA-style joint MAE + contrastive) for the batches
produced by dataset_ssl4eo.py: {s1_a, s2_a, s1_b, s2_b, loc}.

Design (the parts that matter):
  * SENSOR-SPECIFIC STEMS: S1 (2ch) and S2 (12/13ch) have different channel
    counts, so each gets its own patch-embed conv into the shared width, plus a
    learned modality embedding. The transformer encoder is SHARED across sensors.
  * MASKED ENCODER + LIGHT DECODER: standard MAE. Encoder sees only visible
    tokens; a small decoder reconstructs masked patches. Per-modality decoder
    heads (different output channel counts).
  * TOGGLEABLE LOSSES (for ablations):
      - MAE reconstruction (per modality)          [on]
      - cross-MODAL InfoNCE  (S1<->S2, same view)  [on]   CROMA-style
      - cross-SEASON InfoNCE (view_a<->view_b)     [off]  SeCo-style
    Flip use_* flags / weights to run the "what does SAR add / does contrastive
    help" studies.

Transformer blocks / masking are standard boilerplate adapted from the MAE
reference; the stems, fusion, heads, and loss wiring are the custom part.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Boilerplate: transformer block, patchify
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


def patchify(imgs, p):
    """(B,C,H,W) -> (B, L, p*p*C)"""
    B, C, H, W = imgs.shape
    h = w = H // p
    x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, p * p * C)


def random_masking(x, mask_ratio):
    """Per-sample shuffle, keep first (1-r). Returns visible tokens, mask (1=masked), ids_restore."""
    B, L, D = x.shape
    keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :keep]
    # equivalent to the MAE reference's gather(x, 1, ids_keep[...,None].expand(-1,-1,D)):
    # take_along_dim broadcasts the size-1 last dim for us, so no manual expand.
    x_kept = torch.take_along_dim(x, ids_keep.unsqueeze(-1), dim=1)
    mask = torch.ones(B, L, device=x.device)
    mask[:, :keep] = 0
    mask = torch.take_along_dim(mask, ids_restore, dim=1)
    return x_kept, mask, ids_restore


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    img: int = 224
    patch: int = 16
    s1_chans: int = 2
    s2_chans: int = 12
    dim: int = 384          # ViT-S
    depth: int = 12
    heads: int = 6
    dec_dim: int = 192
    dec_depth: int = 4
    dec_heads: int = 6
    proj_dim: int = 128
    mask_ratio: float = 0.75
    temp: float = 0.07
    norm_pix: bool = True
    # loss toggles + weights
    use_mae: bool = True
    use_crossmodal: bool = True
    use_crossseason: bool = False
    w: dict = field(default_factory=lambda: {"mae": 1.0, "cm": 1.0, "cs": 1.0})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultiSensorMAE(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        L = (cfg.img // cfg.patch) ** 2
        self.L, self.p = L, cfg.patch

        # --- sensor-specific stems + modality embeddings (the custom part) ---
        self.stem_s1 = nn.Conv2d(cfg.s1_chans, cfg.dim, cfg.patch, cfg.patch)
        self.stem_s2 = nn.Conv2d(cfg.s2_chans, cfg.dim, cfg.patch, cfg.patch)
        self.mod_s1 = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.mod_s2 = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos = nn.Parameter(torch.zeros(1, L, cfg.dim))

        # --- shared encoder ---
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)

        # --- decoder (shared trunk, per-modality output heads) ---
        self.dec_embed = nn.Linear(cfg.dim, cfg.dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, L, cfg.dec_dim))
        self.dec_blocks = nn.ModuleList([Block(cfg.dec_dim, cfg.dec_heads) for _ in range(cfg.dec_depth)])
        self.dec_norm = nn.LayerNorm(cfg.dec_dim)
        self.head_s1 = nn.Linear(cfg.dec_dim, cfg.patch ** 2 * cfg.s1_chans)
        self.head_s2 = nn.Linear(cfg.dec_dim, cfg.patch ** 2 * cfg.s2_chans)

        # --- contrastive projection heads ---
        self.proj_cm = self._mlp(cfg.dim, cfg.proj_dim)     # cross-modal
        self.proj_view = self._mlp(cfg.dim, cfg.proj_dim)   # cross-season (on fused)

        self._init()

    @staticmethod
    def _mlp(din, dout):
        return nn.Sequential(nn.Linear(din, din), nn.GELU(), nn.Linear(din, dout))

    def _init(self):
        for p in (self.pos, self.dec_pos, self.mod_s1, self.mod_s2, self.mask_token):
            nn.init.normal_(p, std=0.02)
        self.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                   if isinstance(m, nn.Linear) else None)

    def _stem(self, img, which):
        stem, mod = (self.stem_s1, self.mod_s1) if which == "s1" else (self.stem_s2, self.mod_s2)
        x = stem(img).flatten(2).transpose(1, 2)            # (B, L, D)
        return x + self.pos + mod

    def encode(self, img, which, mask_ratio):
        x = self._stem(img, which)
        if mask_ratio > 0:
            x, mask, ids_restore = random_masking(x, mask_ratio)
        else:
            B = x.shape[0]
            mask = torch.zeros(B, self.L, device=x.device)
            ids_restore = torch.arange(self.L, device=x.device).unsqueeze(0).expand(B, -1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = x.mean(dim=1)                                   # pooled global rep
        return x, z, mask, ids_restore

    def decode(self, latent, ids_restore, which):
        x = self.dec_embed(latent)
        B, n, D = x.shape
        pad = self.mask_token.expand(B, self.L - n, -1)
        x = torch.cat([x, pad], dim=1)
        x = torch.take_along_dim(x, ids_restore.unsqueeze(-1), dim=1)
        x = x + self.dec_pos
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        return (self.head_s1 if which == "s1" else self.head_s2)(x)   # (B, L, p*p*C)

    def _mae_loss(self, pred, imgs, mask):
        target = patchify(imgs, self.p)
        if self.cfg.norm_pix:
            mu = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)
            target = (target - mu) / (var + 1e-6).sqrt()
        loss = ((pred - target) ** 2).mean(-1)              # (B, L)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    @staticmethod
    def _info_nce(z1, z2, temp):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / temp
        labels = torch.arange(z1.shape[0], device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, batch):
        cfg = self.cfg
        s1a, s2a = batch["s1_a"], batch["s2_a"]
        lat1, z1, m1, ir1 = self.encode(s1a, "s1", cfg.mask_ratio)
        lat2, z2, m2, ir2 = self.encode(s2a, "s2", cfg.mask_ratio)

        losses = {}
        if cfg.use_mae:
            p1 = self.decode(lat1, ir1, "s1")
            p2 = self.decode(lat2, ir2, "s2")
            losses["mae"] = 0.5 * (self._mae_loss(p1, s1a, m1) + self._mae_loss(p2, s2a, m2))
        if cfg.use_crossmodal:
            losses["cm"] = self._info_nce(self.proj_cm(z1), self.proj_cm(z2), cfg.temp)
        if cfg.use_crossseason:
            _, z1b, _, _ = self.encode(batch["s1_b"], "s1", 0.0)
            _, z2b, _, _ = self.encode(batch["s2_b"], "s2", 0.0)
            za = self.proj_view(0.5 * (z1 + z2))
            zb = self.proj_view(0.5 * (z1b + z2b))
            losses["cs"] = self._info_nce(za, zb, cfg.temp)

        total = sum(cfg.w[k] * v for k, v in losses.items())
        return total, {k: v.detach().item() for k, v in losses.items()}


# --------------------------------------------------------------------------- #
# Self-test: fabricate a loader-shaped batch, forward + backward
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Cfg(use_crossseason=True)          # exercise all three losses
    model = MultiSensorMAE(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.1f}M  (encoder is ViT-S/{cfg.patch})")

    B = 4
    batch = {
        "s1_a": torch.randn(B, cfg.s1_chans, cfg.img, cfg.img),
        "s2_a": torch.randn(B, cfg.s2_chans, cfg.img, cfg.img),
        "s1_b": torch.randn(B, cfg.s1_chans, cfg.img, cfg.img),
        "s2_b": torch.randn(B, cfg.s2_chans, cfg.img, cfg.img),
        "loc": ["a", "b", "c", "d"],
    }
    total, parts = model(batch)
    print("losses:", {k: round(v, 4) for k, v in parts.items()}, "total:", round(total.item(), 4))
    assert torch.isfinite(total), "non-finite loss"
    total.backward()
    grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(grad) and grad > 0, "bad gradients"
    print("forward + backward OK; gradients finite and non-zero.")
    print("Flip cfg.use_* to ablate (e.g. use_crossmodal=False for MAE-only).")