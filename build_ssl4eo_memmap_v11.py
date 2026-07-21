#!/usr/bin/env python3
"""
build_ssl4eo_memmap_v11.py
==========================
Day-1 data prep for SSL4EO-S12 *v1.1* (16-bit, multi-season, WebDataset/Zarr).

Streams the WebDataset shards (local dir or HF http), decodes S1GRD + S2L1C,
and writes co-registered multi-season memmaps + a shared manifest + per-band
stats, in the SAME format make_synthetic_memmap.py produces:

    out/
      s1.dat   (N, 2, H, W) float16   VV, VH         (S1 GRD is 16-bit float)
      s2.dat   (N, C, H, W) uint16    12 or 13 bands (S2 L1C is 16-bit int)
      manifest.json   samples[i]=[loc,season]; groups[loc]=[rows]; per-modality dtype
      stats.npz       s1_mean/std (2,), s2_mean/std (C,)

Rows are flattened (location, season). `groups` lets the loader pull a second
season of the SAME location as a cross-season positive (SeCo-style).

>>> CONFIRM BEFORE A REAL RUN:
>>>   Run with --inspect first. It prints the keys/types/shapes of the first
>>>   shard sample so you can verify decode_sample() matches the real structure.
>>>   Cross-check against the official loaders in
>>>   github.com/DLR-MF-DAS/SSL4EO-S12-v1.1 -- exact Zarr key names can change.
"""
import argparse, io, json, os, sys, tempfile, time
from datetime import datetime, timezone
import numpy as np

try:
    import webdataset as wds
except ImportError:
    sys.exit("pip install webdataset xarray zarr tqdm numpy --break-system-packages")
try:
    import zarr
except ImportError:
    sys.exit("pip install zarr --break-system-packages")
from tqdm import tqdm

S2_BANDS_13 = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
S2_DROP_FOR_12 = {"B10"}
S1_BANDS = ["VV", "VH"]


def select_s2_idx(use_13):
    if use_13:
        return list(range(13)), list(S2_BANDS_13)
    idx = [i for i, b in enumerate(S2_BANDS_13) if b not in S2_DROP_FOR_12]
    return idx, [S2_BANDS_13[i] for i in idx]


# --------------------------------------------------------------------------- #
# DECODE ADAPTER  -- verify with --inspect, cross-check official repo
# --------------------------------------------------------------------------- #
def _open_zarr_zip(raw: bytes) -> np.ndarray:
    """Decode an in-sample .zarr.zip blob to an ndarray (seasons, bands, H, W)."""
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        tmp.write(raw); tmp.flush()
        store = zarr.storage.ZipStore(tmp.name, mode="r")
        node = zarr.open(store, mode="r")
        if hasattr(node, "shape"):                 # node is an array
            arr = node[...]
        else:                                      # node is a group: take first array
            keys = list(node.array_keys())
            if not keys:
                raise ValueError("no array inside zarr group")
            arr = node[keys[0]][...]
        store.close()
    return np.asarray(arr)


def _find_key(sample: dict, modality: str):
    """Find the sample key holding a given modality (e.g. 'S1GRD')."""
    for k in sample:
        if modality.lower() in k.lower() and ("zarr" in k.lower() or "zip" in k.lower()):
            return k
    for k in sample:                               # looser fallback
        if modality.lower() in k.lower():
            return k
    return None


def decode_sample(sample: dict, s2_idx):
    """
    Return (s1, s2):
        s1 (seasons, 2, H, W) float
        s2 (seasons, len(s2_idx), H, W) float
    EDIT this if --inspect shows a different layout (e.g. bands-first, or a
    separate file per season).
    """
    k1, k2 = _find_key(sample, "S1GRD"), _find_key(sample, "S2L1C")
    if k1 is None or k2 is None:
        raise KeyError(f"missing modality keys; have {list(sample)}")
    s1 = _open_zarr_zip(sample[k1])               # expect (seasons, 2, H, W)
    s2 = _open_zarr_zip(sample[k2])               # expect (seasons, 13, H, W)
    if s1.ndim != 4 or s2.ndim != 4:
        raise ValueError(f"expected 4D arrays, got s1{s1.shape} s2{s2.shape}")
    s2 = s2[:, s2_idx]                            # select bands
    return s1.astype(np.float32), s2.astype(np.float32)


def _check_hw(a: np.ndarray, store: int) -> np.ndarray:
    """a: (bands, H, W) -> assert it is already store x store. Never resample:
    v1.1 ships pre-gridded, so a wrong size means a decode bug (wrong band/axis),
    which we want to surface, not silently interpolate over."""
    if a.shape[-1] != store or a.shape[-2] != store:
        raise ValueError(
            f"patch is {a.shape[-2]}x{a.shape[-1]}, expected {store}x{store}; "
            "v1.1 should be pre-gridded -- fix decode_sample() or set --store to match")
    return a


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True,
                    help="brace-expandable WebDataset URL(s): local "
                         "'data/ssl4eo/shard-{000000..000050}.tar' or an https HF url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--locs", type=int, default=20_000, help="number of LOCATIONS")
    ap.add_argument("--store", type=int, default=264)
    ap.add_argument("--use-13-bands", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print first sample's structure and exit")
    args = ap.parse_args()

    s2_idx, s2_bands = select_s2_idx(args.use_13_bands)
    C, store = len(s2_bands), args.store
    os.makedirs(args.out, exist_ok=True)
    ds = wds.WebDataset(args.shards, shardshuffle=False, handler=wds.warn_and_continue)

    if args.inspect:
        for sample in ds:
            print("Sample keys:")
            for k, v in sample.items():
                info = f"{type(v).__name__}, {len(v)} bytes" if isinstance(v, bytes) else type(v).__name__
                print(f"  {k:40s} {info}")
            for mod in ("S1GRD", "S2L1C"):
                k = _find_key(sample, mod)
                if k:
                    try:
                        a = _open_zarr_zip(sample[k])
                        print(f"  -> {mod}: decoded array {a.shape} {a.dtype}")
                    except Exception as e:
                        print(f"  -> {mod}: decode FAILED ({e}) -- edit decode_sample()")
            print("\nIf shapes look like (seasons, bands, H, W), you're good. "
                  "Otherwise adjust decode_sample(). Re-run without --inspect.")
            return

    # peek one sample to learn seasons/H/W, then size the memmaps
    first = next(iter(ds))
    s1_0, s2_0 = decode_sample(first, s2_idx)
    n_seasons = s1_0.shape[0]
    N = args.locs * n_seasons
    print(f"Detected {n_seasons} seasons/location -> allocating {N:,} rows "
          f"({args.locs:,} locs), s2 {C}-band {store}x{store}")

    s1_mm = np.memmap(os.path.join(args.out, "s1.dat"), dtype=np.float16, mode="w+",
                      shape=(N, 2, store, store))
    s2_mm = np.memmap(os.path.join(args.out, "s2.dat"), dtype=np.uint16, mode="w+",
                      shape=(N, C, store, store))

    s1_sum, s1_sq = np.zeros(2), np.zeros(2)
    s2_sum, s2_sq = np.zeros(C), np.zeros(C)
    px = 0
    samples, groups, w, n_loc, skipped = [], {}, 0, 0, 0
    t0 = time.time()
    pbar = tqdm(total=args.locs, desc="locations", unit="loc")

    for li, sample in enumerate(ds):
        if n_loc >= args.locs:
            break
        try:
            s1, s2 = decode_sample(sample, s2_idx)        # (seasons, b, H, W)
        except Exception as e:                            # noqa: BLE001
            skipped += 1
            if skipped <= 10:
                pbar.write(f"  skip sample {li}: {e}")
            continue
        loc = sample.get("__key__", f"loc{n_loc:06d}")
        groups[loc] = []
        for s in range(min(n_seasons, s1.shape[0])):
            if w >= N:
                break
            a1 = _check_hw(s1[s], store).astype(np.float16)
            a2 = np.clip(_check_hw(s2[s], store), 0, 65535).astype(np.uint16)
            s1_mm[w], s2_mm[w] = a1, a2
            a1f = a1.astype(np.float64); a2f = a2.astype(np.float64)
            s1_sum += a1f.reshape(2, -1).sum(1);  s1_sq += (a1f ** 2).reshape(2, -1).sum(1)
            s2_sum += a2f.reshape(C, -1).sum(1);  s2_sq += (a2f ** 2).reshape(C, -1).sum(1)
            px += store * store
            samples.append([loc, f"s{s}"]); groups[loc].append(w); w += 1
        n_loc += 1
        pbar.update(1)
    pbar.close()
    s1_mm.flush(); s2_mm.flush()

    if w < N:                                             # truncate to actual rows
        print(f"  wrote {w:,}/{N:,} rows; truncating")
        del s1_mm, s2_mm
        os.truncate(os.path.join(args.out, "s1.dat"), w * 2 * store * store * 2)   # fp16=2B
        os.truncate(os.path.join(args.out, "s2.dat"), w * C * store * store * 2)   # u16=2B
        N = w

    def mstd(ssum, sq):
        m = ssum / px
        return m.astype(np.float32), np.maximum(np.sqrt(np.maximum(sq / px - m ** 2, 0)), 1e-6).astype(np.float32)
    s1_mean, s1_std = mstd(s1_sum, s1_sq)
    s2_mean, s2_std = mstd(s2_sum, s2_sq)
    np.savez(os.path.join(args.out, "stats.npz"),
             s1_mean=s1_mean, s1_std=s1_std, s2_mean=s2_mean, s2_std=s2_std)

    manifest = {
        "dataset": "SSL4EO-S12 v1.1 multi-sensor memmap",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_written": N, "n_locations": n_loc, "n_seasons": n_seasons,
        "n_skipped": skipped, "store_size": store,
        "s1_dtype": "float16", "s2_dtype": "uint16",
        "s1_bands": S1_BANDS, "s2_bands": s2_bands,
        "s1_memmap": "s1.dat", "s1_shape": [N, 2, store, store],
        "s2_memmap": "s2.dat", "s2_shape": [N, C, store, store],
        "stats_file": "stats.npz",
        "samples": samples, "groups": groups,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    gb = N * (2 * 2 + C * 2) * store * store / 1e9
    print(f"\nDone: {n_loc:,} locs / {N:,} rows, {skipped:,} skipped, "
          f"{(time.time()-t0)/60:.1f} min, ~{gb:.1f} GB")
    verify(args.out)


def verify(out_dir: str, k: int = 8):
    with open(os.path.join(out_dir, "manifest.json")) as f:
        m = json.load(f)
    N, store, C = m["n_written"], m["store_size"], len(m["s2_bands"])
    s1 = np.memmap(os.path.join(out_dir, "s1.dat"), dtype=np.float16, mode="r",
                   shape=(N, 2, store, store))
    s2 = np.memmap(os.path.join(out_dir, "s2.dat"), dtype=np.uint16, mode="r",
                   shape=(N, C, store, store))
    rng = np.random.default_rng(0)
    print(f"\nVerify: N={N:,} locs={m['n_locations']:,} seasons={m['n_seasons']} s2_bands={C}")
    for i in rng.integers(0, N, size=min(k, N)):
        a1, a2 = np.array(s1[i]), np.array(s2[i])
        assert a1.shape == (2, store, store) and a2.shape == (C, store, store)
        assert np.isfinite(a1).all()
        loc, season = m["samples"][i]
        print(f"  [{i:>6}] {loc}/{season}  s1[{a1.min():.1f},{a1.max():.1f}] "
              f"s2[{a2.min()},{a2.max()}]  OK")
    # check a location actually has multiple seasons grouped
    multi = [g for g in m["groups"].values() if len(g) > 1]
    print(f"  {len(multi):,} locations have >1 season (cross-season positives available)")
    print("Eyeball next: render s2 RGB (B4,B3,B2) + s1 VV for one location's seasons.")


if __name__ == "__main__":
    main()
