# GaussianAvatars — SMPLX Extension

This repository extends [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) to support **full-body SMPLX avatars** reconstructed from multi-view video. It adds:

- `train_smplx.py` — training with Gaussians bound to an SMPLX mesh instead of FLAME
- `render_smplx.py` — offline rendering for SMPLX avatars
- `convert_vhap_to_gaussians.py` — converts output from the [adapted VHAP tracker](https://github.com/ShenhanQian/VHAP) to the GaussianAvatars dataset format
- `local_viewer_smplx.py` — interactive viewer for SMPLX avatars
- Helper scripts for the [SignAvatars / How2Sign](https://signavatars.github.io/) and orange multiview datasets

The original FLAME head-only pipeline (`train.py`, `render.py`) is fully preserved and unchanged.

## Acknowledgements

This work is directly built on top of [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) by Qian et al. (CVPR 2024). The original codebase, licenses, and model architecture are reproduced here with minimal modification. All credit for the core Gaussian Avatar method, the FLAME binding approach, and the rendering pipeline belongs to the original authors.

> Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., & Nießner, M. (2024).
> **GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians.**
> CVPR 2024. [[paper]](http://arxiv.org/abs/2312.02069) [[project]](https://shenhanqian.github.io/gaussian-avatars)

The face tracking pipeline (both FLAME and SMPLX) uses [VHAP](https://github.com/ShenhanQian/VHAP).

## Licenses

This work inherits the license of the original GaussianAvatars repository:
[CC-BY-NC-SA-4.0](./LICENSE.md)

> Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

This project also uses [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) under its [original license](./LICENSE_GS.md).
Mesh rendering operations are adapted from [NVDiffRec](https://github.com/NVlabs/nvdiffrec) and [NVDiffRast](https://github.com/NVlabs/nvdiffrast).

---

## Setup

### 1. Installation

See [doc/installation.md](doc/installation.md) for the full installation guide. The short version (Linux, CUDA 11.8):

```shell
git clone <this-repo-url> --recursive
cd GaussianAvatars_SMPLX
bash conda_env.sh           # creates gaussian-avatars env with CUDA 11.8 + PyTorch
conda activate gaussian-avatars
pip install -e smplx/       # bundled SMPLX package
pip install -r requirements.txt
```

Then place the SMPLX model files in `smplx_models/smplx/` (download from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)):
```
smplx_models/smplx/SMPLX_NEUTRAL.npz
smplx_models/smplx/SMPLX_MALE.npz
smplx_models/smplx/SMPLX_FEMALE.npz
```

### 2. Download pre-processed data

See the original [download guide](doc/download.md) for the NeRSemble demo data (FLAME pipeline).

For SMPLX datasets, follow the dataset-specific preparation steps below.

---

## Pipelines

Two independent pipelines are supported. Both produce a `transforms_train.json` / `transforms_val.json` / `transforms_test.json` layout, but use different parametric models and training scripts.

| | FLAME (original) | SMPLX (this extension) |
|---|---|---|
| Tracker | VHAP `track_nersemble.py` | adapted VHAP `track_nersemble_smplx.py` |
| Mesh | head-only, ~5 143 verts | full body, 10 475 verts |
| Param file | `flame_param/*.npz` | `smplx_param/*.npz` |
| Training script | `train.py` | `train_smplx.py` |
| Render script | `render.py` | `render_smplx.py` |
| Viewer | `local_viewer.py` | `local_viewer_smplx.py` |

---

## Dataset Preparation

### A. NeRSemble (FLAME — original pipeline)

The original GaussianAvatars uses pre-processed NeRSemble data exported by VHAP. Follow the [download guide](doc/download.md) to get the demo data, or run VHAP on your own NeRSemble sequences:

```shell
# From the VHAP repository root
python vhap/track_nersemble.py \
  --data.root_folder data/nersemble_v2/ \
  --data.subject 060 \
  --data.sequence EMO-1-shout+laugh \
  --exp.output_folder output/flame/060_EMO-1

python vhap/export_as_nerf_dataset.py \
  --src_folder output/flame/060_EMO-1/<timestamp> \
  --tgt_folder /path/to/GaussianAvatars_SMPLX/data/nersemble_060_flame \
  --background-color white \
  --no-create_mask_from_mesh
```

### B. Orange multiview dataset (SMPLX — adapted VHAP)

The orange multiview dataset is a calibrated multi-camera capture processed through the adapted VHAP SMPLX tracker. The full pipeline is automated by `process_orange_sequences.sh` in the VHAP repository. The steps are:

**1. Run the adapted VHAP pipeline** (from the VHAP repository):

```shell
# Produces: output/smplx/<subject>_<seq>/ and export/orange_multicam/<subject>_<seq>/
bash process_orange_sequences.sh --only-seq EMO-1-shout+laugh
```

This runs background matting, OpenPose + STAR landmark detection, SMPLX tracking, and export automatically.

**2. Convert to GaussianAvatars format:**

```shell
# From GaussianAvatars_SMPLX root
SUBJECT=orange_multicam
SEQ=EMO-1-shout+laugh
VHAP=/path/to/VHAP

TRACKED_PARAMS=$(ls -t ${VHAP}/output/smplx/${SUBJECT}_${SEQ}/*/tracked_flame_params_*.npz | head -1)

conda activate gaussian-avatars
python convert_vhap_to_gaussians.py \
  --vhap_export    ${VHAP}/export/orange_multicam/${SUBJECT}_${SEQ} \
  --tracked_params ${TRACKED_PARAMS} \
  --output         data/${SUBJECT}_${SEQ}_smplx \
  --smplx_models   ./smplx_models \
  --alpha_maps_dir ${VHAP}/data/orange_multicam/${SUBJECT}/sequences/${SEQ}/alpha_maps \
  --raw_images_dir ${VHAP}/data/orange_multicam/${SUBJECT}/sequences/${SEQ}/images_4
```

> **Why `--alpha_maps_dir` and `--raw_images_dir`?**
> The VHAP exporter writes face-masked images to the export folder. These two flags override them with the original full-body images (`images_4/`, 2× downsampled) and the full-body background-matting masks (`alpha_maps/`) so the body region is correctly supervised during training.

### C. SignAvatars / How2Sign dataset (SMPLX)

The [SignAvatars](https://signavatars.github.io/) dataset provides monocular sign-language video from the How2Sign corpus. Preprocessing extracts frames, generates background masks, and merges them into RGBA images.

**1. Extract frames and masks for a signer/video pair:**

```shell
# Usage: bash extract_sequence_signavatars.sh <video_id> <signer_id>
# Example:
bash extract_sequence_signavatars.sh _P01_12 S05
```

This will:
- Extract frames from `raw_videos/<video_id>_*-<signer_id>-rgb_front.mp4`
- Run `vhap/preprocess_video.py` with `robust_video_matting` to generate `alpha_maps/`
- Merge the masks into RGBA `images/` using ffmpeg

**2. Track with the adapted VHAP SMPLX tracker** (from the VHAP repository):

```shell
python vhap/track_nersemble_smplx.py \
  --data.root_folder data/SignAvatars/how2sign/<signer_id>/<video_id> \
  --data.subject <signer_id> \
  --data.sequence <sentence_num> \
  --data.background_color white \
  --data.n_downsample_rgb 4 \
  --pipeline.rgb-init-texture.num-steps 0 \
  --pipeline.rgb-init-all.num-steps 0 \
  --exp.output_folder output/smplx/signavatars_<signer_id>_<video_id>
```

**3. Export and convert:**

```shell
TRACKED_PARAMS=$(ls -t output/smplx/signavatars_<signer_id>_<video_id>/*/tracked_flame_params_*.npz | head -1)

python vhap/export_as_nerf_dataset_mask.py \
  --src_folder $(dirname ${TRACKED_PARAMS}) \
  --tgt_folder /path/to/GaussianAvatars_SMPLX/data/signavatars_<signer_id>_<video_id> \
  --background-color white \
  --no-create_mask_from_mesh

# Then convert (from GaussianAvatars_SMPLX root):
conda activate gaussian-avatars
python convert_vhap_to_gaussians.py \
  --vhap_export  data/signavatars_<signer_id>_<video_id> \
  --tracked_params ${TRACKED_PARAMS} \
  --output       data/signavatars_<signer_id>_<video_id>_converted \
  --smplx_models ./smplx_models
```

---

## Training

### Original FLAME pipeline (head avatars)

```shell
SUBJECT=306

python train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
  --eval --bind_to_mesh --white_background --port 60000
```

<details>
<summary>Command line arguments</summary>

- `-s` / `--source_path` — path to the exported NeRF-style dataset
- `-m` / `--model_path` — output path for the trained model
- `--eval` — enable train/val/test split evaluation
- `--bind_to_mesh` — bind Gaussians to the FLAME mesh
- `--white_background` — use white background
- `--iterations` — total training iterations (default: 30 000)
- `--port` — GUI server port (default: 60 000)

</details>

### SMPLX pipeline (full-body avatars)

```shell
# Orange multiview example
SUBJECT=orange_multicam
SEQ=EMO-1-shout+laugh

python train_smplx.py \
  -s data/${SUBJECT}_${SEQ}_smplx \
  -m output/${SUBJECT}_${SEQ}_smplx \
  --bind_to_mesh --eval --white_background \
  --lambda_xyz 0.05 \
  --port 60000
```

```shell
# SignAvatars example
python train_smplx.py \
  -s data/signavatars_S05_P01_12_converted \
  -m output/signavatars_S05_P01_12 \
  --bind_to_mesh --eval --white_background \
  --port 60000
```

<details>
<summary>Command line arguments (additional to FLAME)</summary>

- `--smplx_models` — path to SMPLX model files (default: `./smplx_models`)
- `--lambda_xyz` — weight for Gaussian position regularisation toward the mesh surface (default: 0.01; use 0.05 for tighter binding)
- `--not_finetune_flame_params` — freeze SMPLX pose/expression parameters during training (recommended to avoid test-frame drift)

</details>

> **Note:** During training, PSNR / SSIM / LPIPS are printed at regular intervals and logged to Tensorboard. Both `train.py` and `train_smplx.py` follow identical evaluation logic.

---

## Rendering

### FLAME

```shell
python render.py \
  -m output/UNION10EMOEXP_306_eval_600k \
  --bind_to_mesh --white_background --eval
```

### SMPLX

```shell
python render_smplx.py \
  -m output/orange_multicam_EMO-1-shout+laugh_smplx \
  --bind_to_mesh --white_background --eval
```

---

## Interactive Viewers

### Remote viewer (during training)

```shell
python remote_viewer.py --port 60000
```

### Local viewer — FLAME

```shell
python local_viewer.py \
  --point_path output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_300000/point_cloud.ply
```

### Local viewer — SMPLX

```shell
python local_viewer_smplx.py \
  --point_path output/orange_multicam_EMO-1-shout+laugh_smplx/point_cloud/iteration_300000/point_cloud.ply
```

---

## Cite

If you use this repository, please cite the original GaussianAvatars paper:

```bibtex
@inproceedings{qian2024gaussianavatars,
  title     = {GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians},
  author    = {Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and
               Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {20299--20309},
  year      = {2024}
}
```

If you use the VHAP tracker, also cite:

```bibtex
@inproceedings{qian2024versatile,
  title     = {Versatile Head Avatars from Portrait Video via Personalized Prior},
  author    = {Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and
               Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle = {arXiv},
  year      = {2024}
}
```
