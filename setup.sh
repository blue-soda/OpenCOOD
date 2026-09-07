#!/usr/bin/env bash

set -euo pipefail

# OpenCOOD/CoBEVFlow setup for Linux/macOS-like shells.
# Defaults create an isolated conda environment named "opencood".
#
# Optional environment variables:
#   CONDA_ENV_NAME=opencood
#   CLONE_FROM_ENV= to clone a known-good conda environment first
#   PYTHON_VERSION=3.7.11
#   TORCH_VARIANT=cu113
#   FORCE_RECREATE=1
#   INSTALL_SPCONV2=0 to skip the default spconv 2.x wheel installation
#   SPCONV2_PACKAGE=spconv-cu113
#   INSTALL_SPCONV121=1 to build legacy spconv 1.2.1 from source
#   COMPILE_CUDA_EXTENSIONS=0 to skip the default bbox CUDA extension build
#   COMPILE_PCDET_EXTENSIONS=1 to build optional FPV-RCNN pcdet CUDA extensions
#   SKIP_PYPCD=1
#   PIP_DEFAULT_TIMEOUT=1000
#   PIP_RETRIES=10
#   INSTALL_CONDA_CUDNN_BOOST=1 to install README's optional conda cudnn/boost packages

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-opencood}"
CLONE_FROM_ENV="${CLONE_FROM_ENV:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.7.11}"
TORCH_VARIANT="${TORCH_VARIANT:-cu113}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
INSTALL_SPCONV2="${INSTALL_SPCONV2:-1}"
SPCONV2_PACKAGE="${SPCONV2_PACKAGE:-spconv-cu113}"
INSTALL_SPCONV121="${INSTALL_SPCONV121:-0}"
COMPILE_CUDA_EXTENSIONS="${COMPILE_CUDA_EXTENSIONS:-1}"
COMPILE_PCDET_EXTENSIONS="${COMPILE_PCDET_EXTENSIONS:-0}"
SKIP_PYPCD="${SKIP_PYPCD:-0}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1000}"
PIP_RETRIES="${PIP_RETRIES:-10}"
INSTALL_CONDA_CUDNN_BOOST="${INSTALL_CONDA_CUDNN_BOOST:-0}"

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
  if [[ -n "$CLONE_FROM_ENV" ]] && conda env list | awk '{print $1}' | grep -qx "$CLONE_FROM_ENV"; then
    log "Cloning known-good environment: $CLONE_FROM_ENV -> $CONDA_ENV_NAME"
    conda create -y -n "$CONDA_ENV_NAME" --clone "$CLONE_FROM_ENV"
    return
  elif [[ -n "$CLONE_FROM_ENV" ]]; then
    warn "Clone source env '$CLONE_FROM_ENV' not found. Creating a fresh environment."
  fi
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip=21.1.2 cmake=3.22.1
}

install_dependencies() {
  if python - <<PY >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.__version__ == "1.10.0+${TORCH_VARIANT}" and torch.cuda.is_available() else 1)
PY
  then
    log "PyTorch CUDA wheel already satisfies torch==1.10.0+${TORCH_VARIANT}; skipping download."
  else
    log "Installing PyTorch wheels using the OpenCDA convention: $TORCH_VARIANT"
    python -m pip install \
      --default-timeout "$PIP_DEFAULT_TIMEOUT" \
      --retries "$PIP_RETRIES" \
      "torch==1.10.0+${TORCH_VARIANT}" \
      "torchvision==0.11.1+${TORCH_VARIANT}" \
      "torchaudio==0.10.0+${TORCH_VARIANT}" \
      -f "https://download.pytorch.org/whl/${TORCH_VARIANT}/torch_stable.html" \
      --no-deps
  fi
  prepend_torch_dll_dir
  if [[ "$INSTALL_CONDA_CUDNN_BOOST" == "1" ]]; then
    log "Installing optional conda cudnn/boost packages"
    conda install -y -n "$CONDA_ENV_NAME" cudnn boost -c conda-forge
  else
    warn "Skipping optional conda cudnn/boost packages. Set INSTALL_CONDA_CUDNN_BOOST=1 to install them."
  fi

  log "Installing Python dependencies"
  python -m pip install --upgrade \
    --default-timeout "$PIP_DEFAULT_TIMEOUT" \
    --retries "$PIP_RETRIES" \
    wheel \
    'setuptools<60' \
    'easydict~=1.9' \
    'numpy>=1.20,<1.22' \
    numba==0.49.0 \
    opencv-python==4.5.2.52 \
    matplotlib==3.4.2 \
    scipy==1.6.3 \
    scikit-image \
    icecream \
    tqdm \
    PyYAML \
    open3d==0.10.0.0 \
    cython \
    tensorboardX \
    shapely==1.8.4 \
    einops \
    h5py==3.8.0 \
    pyquaternion \
    ipdb \
    python-lzf

  if [[ "$INSTALL_SPCONV2" == "1" ]]; then
    log "Installing spconv 2.x wheel package: $SPCONV2_PACKAGE"
    python -m pip install --default-timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" "$SPCONV2_PACKAGE"
    python - <<'PY'
import spconv
import spconv.pytorch as spconv_pt
print("spconv", spconv.__version__, spconv.__file__)
PY
  else
    warn "Skipping spconv 2.x wheel installation. Set INSTALL_SPCONV2=1 to install it."
  fi
}

install_pypcd() {
  if [[ "$SKIP_PYPCD" == "1" ]]; then
    log "Skipping pypcd installation because SKIP_PYPCD=1"
    return
  fi

  log "Installing and patching pypcd for Python 3"
  python -m pip install --default-timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" git+https://github.com/klintan/pypcd.git
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
  python -m pip install --default-timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" -e "$ROOT_DIR" --no-deps

  if [[ "$COMPILE_CUDA_EXTENSIONS" == "1" ]]; then
    log "Building OpenCOOD CUDA/Cython bbox extensions"
    python "$ROOT_DIR/opencood/utils/setup.py" build_ext --inplace
  else
    warn "Skipping OpenCOOD CUDA extension build. Set COMPILE_CUDA_EXTENSIONS=1 to try it."
  fi

  if [[ "$COMPILE_PCDET_EXTENSIONS" == "1" ]]; then
    ensure_cuda_home
    log "Building optional OpenCOOD pcdet extensions"
    python "$ROOT_DIR/opencood/pcdet_utils/setup.py" build_ext --inplace
  else
    warn "Skipping optional pcdet extension build. Set COMPILE_PCDET_EXTENSIONS=1 to try it."
  fi
}

install_spconv121() {
  if [[ "$INSTALL_SPCONV121" != "1" ]]; then
    warn "Skipping legacy spconv 1.2.1 source build. Set INSTALL_SPCONV121=1 to try it."
    return
  fi

  need_cmd git
  ensure_cuda_home

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
  python -m pip install --default-timeout "$PIP_DEFAULT_TIMEOUT" --retries "$PIP_RETRIES" "$wheel"
  popd >/dev/null
}

prepend_torch_dll_dir() {
  if [[ "$(python - <<'PY'
import os
print(os.name)
PY
)" != "nt" ]]; then
    return
  fi
  local torch_lib
  torch_lib="$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"
  if [[ -d "$torch_lib" ]]; then
    export PATH="$torch_lib:$PATH"
    log "Added PyTorch DLL directory to PATH: $torch_lib"
  fi
}

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    return
  fi
  if [[ -n "${CUDA_PATH:-}" ]]; then
    export CUDA_HOME="$CUDA_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_bin
    nvcc_bin="$(dirname "$(command -v nvcc)")"
    CUDA_HOME="$(cd "$nvcc_bin/.." && pwd)"
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH"
    log "Detected CUDA_HOME: $CUDA_HOME"
    return
  fi
  fail "CUDA_HOME is required for pcdet_utils/spconv compilation, but nvcc was not found. PyTorch CUDA runtime can still work without nvcc; compiling CUDA extensions requires the full CUDA Toolkit."
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
    import spconv.pytorch as spconv_pt
    print("spconv ok", spconv.__version__, pathlib.Path(spconv.__file__).resolve())
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
