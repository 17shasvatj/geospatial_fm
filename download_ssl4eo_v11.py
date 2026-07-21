#!/usr/bin/env python3
"""
download_ssl4eo_v11.py
======================
Fetch a BOUNDED subset of SSL4EO-S12 v1.1 from HuggingFace.

Why bounded: the full two-modality set across all ~246k locations is ~700GB-1TB.
You only need ~20-25k locations, so you pull only the first N shards. On RunPod
ingress is free, so re-running this each session costs nothing but a few minutes.

Run this on a CHEAP CPU pod (or at GPU-pod startup) -- it has no GPU work.

Note on variants:
  * WebDataset variant (embed2scale/SSL4EO-S12-v1.1): .tar shards, each bundles
    ALL modalities per sample. You subset by SHARD COUNT (this script's default).
    Modality selection then happens at decode time in the build script.
  * Zarr variant (embed2scale/SSL4EO-S12-v1.1-Zarr): per-modality dirs -> you can
    filter modalities with --allow "*/S1GRD/*" --allow "*/S2L1C/*" instead.
"""
import argparse
import os
import re
import sys

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub tqdm --break-system-packages")
from tqdm import tqdm


def _dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="embed2scale/SSL4EO-S12-v1.1")
    ap.add_argument("--out", default="data/ssl4eo-s12")
    ap.add_argument("--num-shards", type=int, default=50,
                    help="how many .tar shards to pull (controls #locations)")
    ap.add_argument("--modalities", default="S1GRD,S2L1C",
                    help="keep shards whose path contains these (only if shards are per-modality)")
    ap.add_argument("--allow", action="append", default=None,
                    help="Zarr-variant modality filter, e.g. --allow '*/S1GRD/*'. "
                         "If set, uses snapshot_download with these patterns instead of shard slicing.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Zarr-variant path: pattern-filtered snapshot
    if args.allow:
        print(f"snapshot_download {args.repo} with patterns {args.allow}")
        snapshot_download(args.repo, repo_type="dataset", local_dir=args.out,
                          allow_patterns=args.allow)
        print(f"Done. ~{_dir_size_gb(args.out):.1f} GB in {args.out}")
        return

    # WebDataset-variant path: first N .tar shards
    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset")
    tars = sorted(f for f in files if f.endswith(".tar"))
    if not tars:
        sys.exit("No .tar shards found. This looks like the Zarr variant -- "
                 "re-run with --allow '*/S1GRD/*' --allow '*/S2L1C/*'.")

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if any(any(m in t for m in mods) for t in tars):     # shards namespaced by modality
        tars = [t for t in tars if any(m in t for m in mods)]

    subset = tars[: args.num_shards]
    print(f"Downloading {len(subset)} of {len(tars)} shards from {args.repo} -> {args.out}")
    local_paths = []
    for f in tqdm(subset, unit="shard"):
        p = hf_hub_download(args.repo, f, repo_type="dataset", local_dir=args.out)
        local_paths.append(p)

    print(f"\nDone: {len(subset)} shards, ~{_dir_size_gb(args.out):.1f} GB in {args.out}")

    # print the exact --shards arg for the build script
    common = os.path.dirname(os.path.commonprefix(local_paths))
    nums = sorted(int(n) for p in local_paths
                  for n in re.findall(r"(\d{4,})\.tar$", os.path.basename(p)))
    if nums and (nums[-1] - nums[0] + 1) == len(nums):   # contiguous -> brace pattern
        pat = re.sub(r"\d{4,}\.tar$", "", os.path.basename(local_paths[0]))
        width = len(re.findall(r"(\d{4,})\.tar$", os.path.basename(local_paths[0]))[0])
        brace = f"{common}/{pat}{{{nums[0]:0{width}d}..{nums[-1]:0{width}d}}}.tar"
        print(f"\nFeed the build script:\n  --shards \"{brace}\"")
    else:
        print(f"\nShards are in: {common}\n  (pass a brace/glob matching them to --shards)")
    print("\nNext: python build_ssl4eo_memmap_v11.py --shards <above> --out ./data --inspect")


if __name__ == "__main__":
    main()
