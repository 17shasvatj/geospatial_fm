#!/usr/bin/env python3
"""
dataset_ssl4eo.py
=================
PyTorch Dataset over the memmaps written by build_ssl4eo_memmap_v11.py
(and the synthetic make_synthetic_memmap.py -- same manifest contract).

Per location it samples an ANCHOR season and a POSITIVE season (SeCo-style
cross-season positive), so the training loop can form:
  - MAE reconstruction on each view
  - cross-MODAL InfoNCE (S1<->S2, same location/season)   [CROMA-style]
  - cross-SEASON InfoNCE (anchor<->positive, same location)

Key correctness points baked in:
  - memmaps opened lazily PER WORKER (an open mmap doesn't survive fork)
  - each slice copied out of the mmap before use (raw slices are views -> aliasing bugs)
  - S1 and S2 within a view get the SAME crop/flip/rot (co-registration preserved);
    the two views get INDEPENDENT augmentation (good for contrastive)
  - plain crop + flips + rot90 only -- no interpolation on multispectral bands
  - handles float16 S1 / uint16 S2 (real) and uint16/uint16 (synthetic) via manifest dtypes
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SSL4EOMemmap(Dataset):
    def __init__(self, manifest_path: str, crop: int = 224, train: bool = True):
        with open(manifest_path) as f:
            self.m = json.load(f)
        self.root = os.path.dirname(os.path.abspath(manifest_path))
        self.train = train
        self.crop = crop
        self.store = self.m["store_size"]
        assert self.crop <= self.store, f"crop {crop} > stored patch {self.store}"

        # dtype resolution: real build writes s1_dtype/s2_dtype; synthetic writes one 'dtype'
        self.s1_dtype = np.dtype(self.m.get("s1_dtype", self.m.get("dtype", "float16")))
        self.s2_dtype = np.dtype(self.m.get("s2_dtype", self.m.get("dtype", "uint16")))
        self.n_rows = self.m["n_written"]
        self.c1 = len(self.m["s1_bands"])
        self.c2 = len(self.m["s2_bands"])
        self.s1_path = os.path.join(self.root, self.m["s1_memmap"])
        self.s2_path = os.path.join(self.root, self.m["s2_memmap"])

        # location -> list of row indices (its seasons)
        self.groups = self.m["groups"]
        self.locs = list(self.groups.keys())

        st = np.load(os.path.join(self.root, self.m["stats_file"]))
        self.s1_mean = torch.tensor(st["s1_mean"], dtype=torch.float32).view(-1, 1, 1)
        self.s1_std = torch.tensor(st["s1_std"], dtype=torch.float32).view(-1, 1, 1)
        self.s2_mean = torch.tensor(st["s2_mean"], dtype=torch.float32).view(-1, 1, 1)
        self.s2_std = torch.tensor(st["s2_std"], dtype=torch.float32).view(-1, 1, 1)

        self._s1 = self._s2 = None       # opened lazily, per worker

    # --- memmap access ----------------------------------------------------- #
    def _ensure(self):
        if self._s1 is None:
            self._s1 = np.memmap(self.s1_path, dtype=self.s1_dtype, mode="r",
                                 shape=(self.n_rows, self.c1, self.store, self.store))
            self._s2 = np.memmap(self.s2_path, dtype=self.s2_dtype, mode="r",
                                 shape=(self.n_rows, self.c2, self.store, self.store))

    def _read(self, row: int):
        # copy out of the mmap (views alias across workers/tensors)
        return np.array(self._s1[row]), np.array(self._s2[row])

    # --- augmentation: shared geometry across s1/s2, no interpolation ------- #
    def _augment(self, s1: np.ndarray, s2: np.ndarray):
        H, W = s1.shape[-2], s1.shape[-1]
        c = self.crop
        if self.train:
            y, x = random.randint(0, H - c), random.randint(0, W - c)
        else:
            y, x = (H - c) // 2, (W - c) // 2
        s1, s2 = s1[:, y:y + c, x:x + c], s2[:, y:y + c, x:x + c]
        if self.train:
            if random.random() < 0.5:                       # hflip
                s1, s2 = s1[:, :, ::-1], s2[:, :, ::-1]
            if random.random() < 0.5:                       # vflip
                s1, s2 = s1[:, ::-1, :], s2[:, ::-1, :]
            k = random.randint(0, 3)                        # rot90 x k
            if k:
                s1, s2 = np.rot90(s1, k, (1, 2)), np.rot90(s2, k, (1, 2))
        # flips/rot90 produce negative strides; torch needs contiguous, float32
        s1 = torch.from_numpy(np.ascontiguousarray(s1, dtype=np.float32))
        s2 = torch.from_numpy(np.ascontiguousarray(s2, dtype=np.float32))
        return s1, s2

    def _view(self, row: int):
        a1, a2 = self._read(row)
        s1, s2 = self._augment(a1, a2)
        s1 = (s1 - self.s1_mean) / self.s1_std
        s2 = (s2 - self.s2_mean) / self.s2_std
        return s1, s2

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, i: int):
        self._ensure()
        loc = self.locs[i]
        rows = self.groups[loc]
        a = random.choice(rows) if self.train else rows[0]
        if len(rows) > 1:
            b = random.choice([r for r in rows if r != a]) if self.train else rows[1]
        else:
            b = a                                           # single-season fallback
        s1_a, s2_a = self._view(a)
        s1_b, s2_b = self._view(b)
        return {"s1_a": s1_a, "s2_a": s2_a, "s1_b": s1_b, "s2_b": s2_b, "loc": loc}


def make_loader(dataset: SSL4EOMemmap, batch_size: int = 128,
                num_workers: int = 8, train: bool = True) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=num_workers, pin_memory=True,           # pin for fast host->GPU copy
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,     # keep the GPU fed
    )


if __name__ == "__main__":
    # quick self-check against whatever memmap dir you pass
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing manifest.json")
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    ds = SSL4EOMemmap(os.path.join(a.data, "manifest.json"), crop=a.crop, train=True)
    print(f"locations={len(ds)}  s1={ds.c1}ch/{ds.s1_dtype}  s2={ds.c2}ch/{ds.s2_dtype}  store={ds.store}")
    dl = make_loader(ds, batch_size=a.batch, num_workers=a.workers)
    t0 = time.time()
    b = next(iter(dl))
    print(f"batch in {time.time()-t0:.2f}s:")
    for k in ("s1_a", "s2_a", "s1_b", "s2_b"):
        v = b[k]
        print(f"  {k}: {tuple(v.shape)} {v.dtype}  mean={v.mean():.3f} std={v.std():.3f}")
    print(f"  locs: {b['loc'][:4]}")
    print("If s1_a/s1_b differ (different seasons) and shapes are "
          f"(B,C,{a.crop},{a.crop}), the loader is wired correctly.")
