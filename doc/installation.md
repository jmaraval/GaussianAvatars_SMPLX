## Requirements

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 11 GB VRAM (we used RTX 2080Ti)

### Software Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (GCC on Linux, Visual Studio on Windows)
- CUDA SDK for PyTorch extensions, installed *after* the C++ compiler
- FFMPEG to create result videos

### Additional Python Packages

- RoMa (rotation representations)
- DearPyGUI (viewer interface)
- NVDiffRast (mesh rendering in viewer)
- SMPLX (full-body parametric model — bundled as `smplx/` in this repo)

### Tested Platforms

| PyTorch Version | CUDA version | Linux |
|-| - | - |
| 2.0.1  | 11.7.1 | Pass (upstream GaussianAvatars) |
| 2.2.0  | 12.1.1 | Pass (upstream GaussianAvatars) |
| **2.5.1** | **11.8** | **Pass (this repo — recommended)** |

---

## Installation

The recommended method uses the `conda_env.sh` script which reproduces the exact environment this repository was developed and tested with (**CUDA 11.8**).

### 1. Clone and create the conda environment

```shell
git clone <this-repo-url> --recursive
cd GaussianAvatars_SMPLX

# Creates the 'gaussian-avatars' conda env with CUDA 11.8 and PyTorch
bash conda_env.sh
conda activate gaussian-avatars
```

`conda_env.sh` performs the following steps automatically:
- Creates a Python 3.10 environment named `gaussian-avatars`
- Installs `cuda-toolkit=11.8.0` from the nvidia conda channel
- Sets `CUDA_HOME` and creates the `lib64` symlink required for CUDA compilation on Linux
- Installs PyTorch with `pytorch-cuda=11.8` via the pytorch conda channel

### 2. Install Python packages

```shell
# Install the bundled SMPLX package from source
pip install -e smplx/

# Install remaining packages (compiles diff-gaussian-rasterization, simple-knn, nvdiffrast — takes a few minutes)
pip install -r requirements.txt
```

### 3. Download SMPLX model files

Register and download the SMPLX model (`.npz` format) from the [SMPLX project page](https://smpl-x.is.tue.mpg.de/), then place the files as follows:

```
smplx_models/
└── smplx/
    ├── SMPLX_NEUTRAL.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_FEMALE.npz
```

---

## Manual installation (alternative to conda_env.sh)

If you prefer to install manually or use a different CUDA version, follow these steps.

### Step 1 — Create conda environment and install CUDA

```shell
conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars

# Install CUDA toolkit (adjust version as needed)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit=11.8.0 ninja
```

### Step 2 — Set up paths (Linux)

```shell
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # avoid "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX

# Re-activate to apply the env variable
conda deactivate && conda activate gaussian-avatars
```

### Step 3 — Install PyTorch and packages

```shell
# Install PyTorch matching your CUDA version (tested: 2.5.1 + CUDA 11.8)
conda install pytorch=2.5.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the bundled SMPLX package
pip install -e smplx/

# Install remaining packages
pip install -r requirements.txt
```
