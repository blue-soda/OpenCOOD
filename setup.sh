#!/usr/bin/env bash

set -euo pipefail

# OpenCOOD/CoBEVFlow setup for Linux/macOS-like shells.
# Defaults create an isolated conda environment named "opencood".
#
# Optional environment variables:
#   CONDA_ENV_NAME=opencood
#   PYTHON_VERSION=3.7.11
#   FORCE_RECREATE=1
#   INSTALL_SPCONV121=1
#   COMPILE_CUDA_EXTENSIONS=1
#   SKIP_PYPCD=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-opencood}"
PYTHON_VERSION="${PYTHON_VERSION:-3.7.11}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
INSTALL_SPCONV121="${INSTALL_SPCONV121:-0}"
COMPILE_CUDA_EXTENSIONS="${COMPILE_CUDA_EXTENSIONS:-0}"
SKIP_PYPCD="${SKIP_PYPCD:-0}"

log() {
  echo "[setup] $*"
}

warn() {
  echo "[setup][warn] $*" >&2
}

fail() {
  echo "[setup][error] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

activate_conda() {
  need_cmd conda
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME" || fail "Failed to activate conda environment: $CONDA_ENV_NAME"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"
}

ensure_env() {
  need_cmd conda
  log "Repository: $ROOT_DIR"
  log "Target conda env: $CONDA_ENV_NAME"

  if env_exists; then
    if [[ "$FORCE_RECREATE" == "1" ]]; then
      log "Removing existing environment: $CONDA_ENV_NAME"
      conda env remove -n "$CONDA_ENV_NAME" -y
    else
      log "Conda environment already exists. Skipping creation."
      return
    fi
  fi

  log "Creating conda environment: $CONDA_ENV_NAME"
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip=21.1.2 cmake=3.22.1
}

install_dependencies() {
  log "Installing PyTorch 1.10.1 with CUDA 11.3 runtime"
  conda install -y -n "$CONDA_ENV_NAME" \
    pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 \
    -c pytorch -c conda-forge
  conda install -y -n "$CONDA_ENV_NAME" cudnn boost -c conda-forge

  log "Installing Python dependencies"
  python -m pip install --upgrade \
    wheel \
    'setuptools<60' \
    'easydict~=1.9' \
    numpy==1.19.5 \
    numba==0.49.0 \
    opencv-python==4.5.5.62 \
    matplotlib==3.3.4 \
    scipy==1.5.4 \
    scikit-image==0.18.3 \
    icecream \
    tqdm \
    PyYAML \
    open3d==0.13.0 \
    cython \
    tensorboardX \
    shapely==1.8.5.post1 \
    einops \
    h5py==3.8.0 \
    pyquaternion \
    ipdb \
    python-lzf
}

install_pypcd() {
  if [[ "$SKIP_PYPCD" == "1" ]]; then
    log "Skipping pypcd installation because SKIP_PYPCD=1"
    return
  fi

  log "Installing and patching pypcd for Python 3"
  python -m pip install git+https://github.com/klintan/pypcd.git
  python - <<'PY'
import pathlib
import pypcd

target = pathlib.Path(pypcd.__file__).resolve().parent / "pypcd.py"
text = target.read_text(encoding="utf-8")
text = text.replace(
    "from cStringIO import StringIO",
    "try:\n    from cStringIO import StringIO\nexcept ImportError:\n    from io import StringIO",
)
if "xrange" in text and "xrange = range" not in text:
    text = text.replace(
        "import numpy as np\n",
        "import numpy as np\n\ntry:\n    xrange\nexcept NameError:\n    xrange = range\n",
    )
target.write_text(text, encoding="utf-8")
print("patched", target)
PY
}

install_opencood() {
  log "Installing OpenCOOD in editable mode"
  python -m pip install -e "$ROOT_DIR"

  if [[ "$COMPILE_CUDA_EXTENSIONS" == "1" ]]; then
    log "Building OpenCOOD CUDA/Cython extensions"
    python "$ROOT_DIR/opencood/utils/setup.py" build_ext --inplace
    python "$ROOT_DIR/opencood/pcdet_utils/setup.py" build_ext --inplace
  else
    warn "Skipping OpenCOOD CUDA extension build. Set COMPILE_CUDA_EXTENSIONS=1 to try it."
  fi
}

install_spconv121() {
  if [[ "$INSTALL_SPCONV121" != "1" ]]; then
    warn "Skipping spconv 1.2.1 build. Set INSTALL_SPCONV121=1 to try it."
    return
  fi

  need_cmd git
  need_cmd nvcc

  local deps_dir="$ROOT_DIR/.deps"
  local spconv_dir="$deps_dir/spconv-v1.2.1"
  mkdir -p "$deps_dir"

  if [[ ! -d "$spconv_dir" ]]; then
    git clone https://github.com/traveller59/spconv.git "$spconv_dir"
  fi

  pushd "$spconv_dir" >/dev/null
  git checkout v1.2.1
  git submodule update --init --recursive
  python setup.py bdist_wheel
  local wheel
  wheel="$(find "$spconv_dir/dist" -maxdepth 1 -name 'spconv-1.2.1-*.whl' -print | sort | tail -n 1)"
  [[ -n "$wheel" ]] || fail "spconv wheel was not generated."
  python -m pip install "$wheel"
  popd >/dev/null
}

run_diagnostics() {
  log "Running import diagnostics"
  python - <<'PY'
import pathlib
import sys
import torch
import opencood
from opencood.version import __version__

print("python", sys.version)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("opencood", __version__, pathlib.Path(opencood.__file__).resolve())
try:
    import pypcd
    print("pypcd ok", pathlib.Path(pypcd.__file__).resolve())
except Exception as exc:
    print("pypcd failed", repr(exc))
try:
    import spconv
    print("spconv ok", pathlib.Path(spconv.__file__).resolve())
except Exception as exc:
    print("spconv unavailable; patched OpenCOOD will use NumPy voxelization fallback:", repr(exc))
PY
}

main() {
  ensure_env
  activate_conda
  log "Python runtime: $(python --version 2>&1)"
  install_dependencies
  install_pypcd
  install_opencood
  install_spconv121
  run_diagnostics

  log "Setup completed."
  log "To use:"
  log "  conda activate $CONDA_ENV_NAME"
  log "  cd \"$ROOT_DIR\""
  log "  export PYTHONPATH=\"$ROOT_DIR\""
  log "  export PYTHONUTF8=1"
}

main "$@"
