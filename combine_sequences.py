#!/usr/bin/env python3
"""
Combine per-sequence GaussianAvatars datasets into a single UNION dataset.

Each per-sequence dataset was produced by convert_vhap_to_gaussians.py and
contains transforms_train.json / transforms_val.json / transforms_test.json
with absolute flame_param_path entries pointing to its own smplx_param/ dir.

The UNION dataset:
  • transforms_train.json — train frames from all --train_datasets, globally
                            unique timestep_index values
  • transforms_val.json  — val frames (held-out camera) from all train datasets,
                            matching global timestep_index values
  • transforms_test.json — test frames from --test_dataset (self-reenactment),
                            timestep indices continuing after all train indices

No image or smplx_param files are copied; all flame_param_path entries keep
their original absolute paths which Python's os.path.join handles correctly.

Usage:
  conda activate gaussian-avatars
  python combine_sequences.py \\
    --train_datasets  data/orange_multicam_EMO-1_smplx \\
                      data/orange_multicam_EMO-2_smplx \\
                      ...  \\
    --test_dataset    data/orange_multicam_FREE_smplx \\
    --output          data/orange_multicam_union_smplx

Then train normally:
  python train_smplx.py -s data/orange_multicam_union_smplx -m output/union ...
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_transforms(dataset_dir: Path, split: str) -> dict:
    p = dataset_dir / f"transforms_{split}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def remap_timesteps(frames: list, offset: int) -> tuple[list, int]:
    """
    Return (remapped_frames, new_offset) where each frame's timestep_index
    is shifted by `offset`. Returns the next available offset.
    """
    if not frames:
        return [], offset

    # Collect unique original timestep indices in order of first appearance
    seen = {}
    for fr in frames:
        t = fr.get("timestep_index")
        if t is not None and t not in seen:
            seen[t] = offset + len(seen)

    remapped = []
    for fr in frames:
        fr2 = dict(fr)
        t = fr2.get("timestep_index")
        if t is not None:
            fr2["timestep_index"] = seen[t]
        remapped.append(fr2)

    next_offset = offset + len(seen)
    return remapped, next_offset


def merge_headers(all_headers: list[dict]) -> dict:
    """
    Take the union of top-level scalar/list fields from all sequence headers.
    Per-frame fields live inside 'frames'; everything else is a header field.
    For scalars (fl_x, cx, h, w, ...) we assert all sequences agree (same
    cameras), then take the first value.  timestep_indices and camera_indices
    are rebuilt from the combined frames later.
    """
    SKIP = {"frames", "timestep_indices", "camera_indices"}
    merged = {}
    for h in all_headers:
        for k, v in h.items():
            if k in SKIP:
                continue
            if k not in merged:
                merged[k] = v
            else:
                if merged[k] != v:
                    print(f"  WARNING: header field '{k}' differs across sequences "
                          f"({merged[k]} vs {v}). Using first value.", file=sys.stderr)
    return merged


def build_combined(frames: list) -> dict:
    """Rebuild timestep_indices and camera_indices from a frame list."""
    ts_set = set()
    cam_set = set()
    for fr in frames:
        if "timestep_index" in fr:
            ts_set.add(fr["timestep_index"])
        if "camera_index" in fr:
            cam_set.add(fr["camera_index"])
    result = {
        "timestep_indices": sorted(ts_set),
        "camera_indices":   sorted(cam_set),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build a UNION GaussianAvatars dataset from multiple per-sequence datasets."
    )
    parser.add_argument(
        "--train_datasets", nargs="+", required=True,
        help="Paths to per-sequence GaussianAvatars datasets used for training."
    )
    parser.add_argument(
        "--test_dataset", required=True,
        help="Path to held-out sequence dataset (self-reenactment test, e.g. FREE)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the UNION dataset."
    )
    parser.add_argument(
        "--val_camera_id", default=None,
        help="Camera ID to use as held-out novel-view camera (default: use the val split "
             "already defined in each sequence's transforms_val.json)."
    )
    args = parser.parse_args()

    train_dirs = [Path(d).resolve() for d in args.train_datasets]
    test_dir   = Path(args.test_dataset).resolve()
    out_dir    = Path(args.output).resolve()

    # ── Validate inputs ────────────────────────────────────────────────────────
    missing = [d for d in train_dirs + [test_dir] if not d.is_dir()]
    if missing:
        for m in missing:
            print(f"ERROR: dataset directory not found: {m}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building UNION dataset → {out_dir}")
    print(f"  Train sequences : {len(train_dirs)}")
    for d in train_dirs:
        print(f"    {d.name}")
    print(f"  Test sequence   : {test_dir.name}")
    print()

    # ── Combine train and val splits ───────────────────────────────────────────
    all_train_frames = []
    all_val_frames   = []
    all_train_headers = []
    offset = 0

    for seq_dir in train_dirs:
        train_data = load_transforms(seq_dir, "train")
        val_data   = load_transforms(seq_dir, "val")

        if train_data is None:
            print(f"  WARNING: no transforms_train.json in {seq_dir.name}, skipping.")
            continue

        train_frames = train_data.get("frames", [])
        val_frames   = val_data.get("frames", []) if val_data else []

        # Determine unique timestep indices in this sequence (from train split)
        local_ts = sorted({fr["timestep_index"] for fr in train_frames
                           if "timestep_index" in fr})
        n_ts = len(local_ts)

        # Build a mapping: local_ts[i] → offset + i
        ts_map = {t: offset + i for i, t in enumerate(local_ts)}

        def remap(frames, ts_map):
            out = []
            for fr in frames:
                fr2 = dict(fr)
                t = fr2.get("timestep_index")
                if t is not None and t in ts_map:
                    fr2["timestep_index"] = ts_map[t]
                out.append(fr2)
            return out

        remapped_train = remap(train_frames, ts_map)
        remapped_val   = remap(val_frames,   ts_map)

        all_train_frames.extend(remapped_train)
        all_val_frames.extend(remapped_val)
        all_train_headers.append(train_data)

        print(f"  {seq_dir.name}: {n_ts} timesteps → global [{offset}, {offset + n_ts - 1}]")
        offset += n_ts

    total_train_ts = offset

    # ── Test split (self-reenactment from FREE sequence) ───────────────────────
    test_train  = load_transforms(test_dir, "train")
    test_val    = load_transforms(test_dir, "val")
    test_test   = load_transforms(test_dir, "test")

    # Use whichever split has frames; prefer test > train for self-reenactment
    test_source = test_test or test_train
    if test_source is None:
        print(f"ERROR: no transforms JSON found in {test_dir}", file=sys.stderr)
        sys.exit(1)

    test_frames_raw = test_source.get("frames", [])
    local_ts_test = sorted({fr["timestep_index"] for fr in test_frames_raw
                             if "timestep_index" in fr})
    n_ts_test = len(local_ts_test)
    ts_map_test = {t: total_train_ts + i for i, t in enumerate(local_ts_test)}

    def remap_test(frames):
        out = []
        for fr in frames:
            fr2 = dict(fr)
            t = fr2.get("timestep_index")
            if t is not None and t in ts_map_test:
                fr2["timestep_index"] = ts_map_test[t]
            out.append(fr2)
        return out

    test_frames = remap_test(test_frames_raw)
    print(f"  {test_dir.name} [TEST]: {n_ts_test} timesteps → global "
          f"[{total_train_ts}, {total_train_ts + n_ts_test - 1}]")
    print()

    # ── Build output JSON files ────────────────────────────────────────────────
    header = merge_headers(all_train_headers)

    def write_split(split_name: str, frames: list, source_headers: list):
        if not frames:
            print(f"  WARNING: no frames for {split_name} split, skipping.")
            return
        combined = build_combined(frames)
        doc = dict(header)        # shared camera intrinsics etc.
        doc.update(combined)      # override timestep_indices / camera_indices
        doc["frames"] = frames
        out_path = out_dir / f"transforms_{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  Wrote {out_path.name}: {len(frames)} frames, "
              f"{len(combined['timestep_indices'])} timesteps, "
              f"{len(combined['camera_indices'])} cameras")

    write_split("train", all_train_frames,  all_train_headers)
    write_split("val",   all_val_frames,    all_train_headers)
    write_split("test",  test_frames,       [test_source])

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("UNION dataset ready.")
    print(f"  {out_dir}/")
    print(f"    transforms_train.json  ({len(all_train_frames)} frames)")
    print(f"    transforms_val.json    ({len(all_val_frames)} frames)")
    print(f"    transforms_test.json   ({len(test_frames)} frames)")
    print()
    print("Train with:")
    print(f"  python train_smplx.py \\")
    print(f"    -s {out_dir} \\")
    print(f"    -m output/smplx_{out_dir.name} \\")
    print(f"    --bind_to_mesh --eval --white_background --lambda_xyz 0.05 --not_finetune_flame_params")
    print()
    print("Render / evaluate:")
    print(f"  python render_smplx.py -m output/smplx_{out_dir.name} --bind_to_mesh --white_background --eval")


if __name__ == "__main__":
    main()
