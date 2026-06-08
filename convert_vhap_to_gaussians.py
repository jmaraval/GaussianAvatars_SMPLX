#!/usr/bin/env python3
"""
Convert VHAP SMPLX tracking output to GaussianAvatars DynamicNerf format.

Usage (from GaussianAvatars root):
  conda activate gaussian-avatars
  python convert_vhap_to_gaussians.py \
    --vhap_export /home/jmaraval/Documents/VHAP/export/orange_multicam_nocam32_bg/multicam_nocam32_bg \
    --tracked_params /home/jmaraval/Documents/VHAP/output/smplx/multicam_nocam32_bg/2026-05-29_18-14-53/tracked_flame_params_30.npz \
    --output ./data/multicam_nocam32_bg_smplx

Then train:
  python train_smplx.py \
    -s ./data/multicam_nocam32_bg_smplx \
    -m ./output/multicam_nocam32_bg_smplx \
    --bind_to_mesh --eval
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def expand_hand_pca(pca_poses, components, mean):
    """Expand (T, n_pca) PCA hand poses to (T, 45) full joint axis-angles."""
    if hasattr(components, "cpu"):
        components = components.detach().cpu().numpy()
    if hasattr(mean, "cpu"):
        mean = mean.detach().cpu().numpy()
    return (pca_poses @ components + mean).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert VHAP SMPLX tracked params to GaussianAvatars dataset format"
    )
    parser.add_argument(
        "--vhap_export", required=True,
        help="VHAP export directory containing transforms_train.json and images/"
    )
    parser.add_argument(
        "--tracked_params", required=True,
        help="Path to tracked_flame_params_NN.npz (use the highest iteration, e.g. _30.npz)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the GaussianAvatars dataset"
    )
    parser.add_argument(
        "--smplx_models", default="./smplx_models",
        help="Path to SMPLX model files (default: ./smplx_models)"
    )
    parser.add_argument(
        "--alpha_maps_dir", default=None,
        help="Path to the raw full-body alpha maps (e.g. .../sequences/01/alpha_maps). "
             "Files must be named cam_<camera_id>_<timestep_id>.<ext>. "
             "Replaces the face-only fg_masks that export_as_nerf_dataset_mask.py generates."
    )
    parser.add_argument(
        "--raw_images_dir", default=None,
        help="Path to the raw (un-masked) images directory (e.g. .../sequences/01/images_4). "
             "Files must be named cam_<camera_id>_<timestep_id>.<ext>. "
             "IMPORTANT: export_as_nerf_dataset_mask.py overwrites the exported images with a "
             "face-only composite. Use this flag to point at the original full-body images so "
             "that the body region has real appearance instead of white background."
    )
    args = parser.parse_args()

    vhap_export    = Path(args.vhap_export).resolve()
    output_dir     = Path(args.output).resolve()
    param_dir      = output_dir / "smplx_param"
    alpha_maps_dir = Path(args.alpha_maps_dir).resolve() if args.alpha_maps_dir else None
    raw_images_dir = Path(args.raw_images_dir).resolve() if args.raw_images_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(exist_ok=True)

    if alpha_maps_dir:
        _alpha_exts = {p.suffix for p in alpha_maps_dir.iterdir() if p.is_file()}
        print(f"   Using full-body alpha maps from: {alpha_maps_dir}  (exts: {_alpha_exts})")
    if raw_images_dir:
        _img_exts = {p.suffix for p in raw_images_dir.iterdir() if p.is_file()}
        print(f"   Using raw (unmasked) images from: {raw_images_dir}  (exts: {_img_exts})")
    else:
        print("   WARNING: --raw_images_dir not set. Using VHAP exported images, which have "
              "the face-only composite applied (body region is white). "
              "Pass --raw_images_dir .../images_4 for correct full-body training.")

    # ----------------------------------------------------------------
    # 1. Load tracked SMPLX parameters
    # ----------------------------------------------------------------
    print(f"[1/5] Loading tracked params: {args.tracked_params}")
    tp = dict(np.load(args.tracked_params, allow_pickle=True))

    T         = tp["rotation"].shape[0]
    n_betas   = int(tp["shape"].shape[0])
    n_expr    = int(tp["expr"].shape[1])
    n_hand    = int(tp["left_hand_pose"].shape[1])  # typically 6 (PCA comps)
    n_verts   = int(tp["static_offset"].shape[1])   # typically 10475 (SMPLX)

    print(f"   T={T}  n_betas={n_betas}  n_expr={n_expr}  "
          f"n_hand_pca={n_hand}  n_verts={n_verts}")

    # VHAP's export_as_nerf_dataset.py relocates the body to the origin by
    # subtracting T_mean from all frame translations AND applying the same
    # shift to every camera matrix in transforms_*.json.  Our converter must
    # apply the identical shift so that SMPLX body positions are consistent
    # with the already-relocated camera poses.
    T_mean = tp["translation"].mean(axis=0).astype(np.float32)
    tp["translation"] = (tp["translation"] - T_mean).astype(np.float32)
    print(f"   T_mean (relocation offset): {T_mean.round(4)}")

    # ----------------------------------------------------------------
    # 2. Expand hand PCA → full 45-dim poses
    # ----------------------------------------------------------------
    print(f"[2/5] Expanding hand PCA ({n_hand}) → full (45) using SMPLX model ...")
    from smplx import SMPLX
    import torch
    smplx_model = SMPLX(
        args.smplx_models,
        num_betas=n_betas,
        num_expression_coeffs=n_expr,
        use_pca=True,
        num_pca_comps=n_hand,
        batch_size=1,
    )
    left_hand_full  = expand_hand_pca(
        tp["left_hand_pose"],
        smplx_model.left_hand_components,
        smplx_model.left_hand_mean,
    )
    right_hand_full = expand_hand_pca(
        tp["right_hand_pose"],
        smplx_model.right_hand_components,
        smplx_model.right_hand_mean,
    )
    del smplx_model
    print(f"   left_hand_full: {left_hand_full.shape}  right_hand_full: {right_hand_full.shape}")

    # ----------------------------------------------------------------
    # 3. Write per-frame SMPLX param npz files
    # ----------------------------------------------------------------
    print(f"[3/5] Writing {T} per-frame npz files to {param_dir}/")
    betas         = tp["shape"].astype(np.float32)          # (n_betas,)
    static_offset = tp["static_offset"][0].astype(np.float32)  # (n_verts, 3)

    for i in range(T):
        frame_id = str(tp["timestep_id"][i])  # e.g. "000042"
        out_path = param_dir / f"{frame_id}.npz"

        data = dict(
            global_orient   = tp["rotation"][i].astype(np.float32),    # (3,)
            body_pose       = tp["body_pose"][i].astype(np.float32),    # (63,)
            jaw_pose        = tp["jaw_pose"][i].astype(np.float32),     # (3,)
            leye_pose       = tp["leye_pose"][i].astype(np.float32),    # (3,)
            reye_pose       = tp["reye_pose"][i].astype(np.float32),    # (3,)
            left_hand_pose  = left_hand_full[i],                        # (45,)
            right_hand_pose = right_hand_full[i],                       # (45,)
            expression      = tp["expr"][i].astype(np.float32),         # (n_expr,)
            translation     = tp["translation"][i].astype(np.float32),  # (3,)
        )
        # betas and static_offset are shared; put them in frame 0 so
        # SMPLXGaussianModel.load_meshes can pick them up from meshes[0]
        if i == 0:
            data["betas"]         = betas
            data["static_offset"] = static_offset

        np.savez(str(out_path), **data)

    print("   done.")

    # ----------------------------------------------------------------
    # 4. Write dummy marker so Scene.__init__ selects DynamicNerf loader
    # ----------------------------------------------------------------
    np.savez(str(output_dir / "canonical_flame_param.npz"))
    print("[4/5] Wrote canonical_flame_param.npz (DynamicNerf marker).")

    # ----------------------------------------------------------------
    # 5. Update transforms JSON (absolute image paths + SMPLX param paths)
    # ----------------------------------------------------------------
    print("[5/5] Writing transforms JSON files ...")
    for split in ["train", "val", "test"]:
        src = vhap_export / f"transforms_{split}.json"
        if not src.exists():
            print(f"   Skipping {split} (file not found)")
            continue

        with open(src) as f:
            transforms = json.load(f)

        for frame in transforms["frames"]:
            # --- image path ---
            # Priority 1: raw unmasked images (before MaskFromFLAME overwrote them)
            # Priority 2: VHAP exported images (face-only composite — wrong for full body)
            if raw_images_dir:
                cam_id      = str(frame.get("camera_id", ""))
                frame_id_str = str(frame.get("timestep_id", ""))
                raw_img_path = None
                for ext in _img_exts:
                    candidate = raw_images_dir / f"cam_{cam_id}_{frame_id_str}{ext}"
                    if candidate.exists():
                        raw_img_path = str(candidate)
                        break
                frame["file_path"] = raw_img_path or str(vhap_export / frame["file_path"])
            else:
                rel_img = frame["file_path"]
                if not os.path.isabs(rel_img):
                    frame["file_path"] = str(vhap_export / rel_img)

            # Resolve fg_mask_path:
            #   Priority 1 — raw full-body alpha maps (--alpha_maps_dir)
            #   Priority 2 — VHAP export fg_masks (face-only, use only as fallback)
            frame["fg_mask_path"] = None
            if alpha_maps_dir:
                cam_id   = str(frame.get("camera_id", ""))
                frame_id_str = str(frame.get("timestep_id", ""))
                for ext in _alpha_exts:
                    candidate = alpha_maps_dir / f"cam_{cam_id}_{frame_id_str}{ext}"
                    if candidate.exists():
                        frame["fg_mask_path"] = str(candidate)
                        break
            if frame["fg_mask_path"] is None and "fg_mask_path" in frame:
                # fallback: VHAP export mask (face-only)
                rel_mask = frame.get("fg_mask_path") or ""
                if rel_mask and not os.path.isabs(rel_mask):
                    rel_mask = str(vhap_export / rel_mask)
                frame["fg_mask_path"] = rel_mask if rel_mask and os.path.exists(rel_mask) else None

            # Replace flame_param_path with our SMPLX param file
            frame_id = str(frame["timestep_id"])
            frame["flame_param_path"] = str(param_dir / f"{frame_id}.npz")

        dst = output_dir / f"transforms_{split}.json"
        with open(dst, "w") as f:
            json.dump(transforms, f, indent=2)
        print(f"   Wrote {dst}  ({len(transforms['frames'])} frames)")

    # ----------------------------------------------------------------
    # Done — print training command
    # ----------------------------------------------------------------
    print()
    print("=" * 65)
    print("Conversion complete.")
    print(f"  n_betas   = {n_betas}")
    print(f"  n_expr    = {n_expr}")
    print(f"  n_verts   = {n_verts}")
    print()
    print("Train with (from GaussianAvatars root):")
    print(f"  conda activate gaussian-avatars")
    print(f"  python train_smplx.py \\")
    print(f"    -s {output_dir} \\")
    print(f"    -m ./output/smplx_multicam_nocam32_bg \\")
    print(f"    --bind_to_mesh \\")
    print(f"    --eval")
    print()
    print("Render with:")
    print(f"  python render_smplx.py \\")
    print(f"    -m ./output/smplx_multicam_nocam32_bg \\")
    print(f"    --bind_to_mesh")
    print("=" * 65)


if __name__ == "__main__":
    main()
