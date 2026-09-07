@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem OpenCOOD/CoBEVFlow setup for Windows.
rem Defaults are chosen for an isolated local test environment named "opencood".
rem Optional environment variables:
rem   CONDA_ENV_NAME=opencood
rem   PYTHON_VERSION=3.7.11
rem   TORCH_VARIANT=cu113
rem   FORCE_RECREATE=1
rem   INSTALL_SPCONV121=1 to try building spconv 1.2.1
rem   COMPILE_CUDA_EXTENSIONS=0 to skip the default bbox CUDA extension build
rem   COMPILE_PCDET_EXTENSIONS=1 to build optional pcdet extensions
rem   SKIP_PYPCD=1

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%" || exit /b 1

if "%CONDA_ENV_NAME%"=="" set "CONDA_ENV_NAME=opencood"
if "%PYTHON_VERSION%"=="" set "PYTHON_VERSION=3.7.11"
if "%TORCH_VARIANT%"=="" set "TORCH_VARIANT=cu113"
if "%FORCE_RECREATE%"=="" set "FORCE_RECREATE=0"
if "%INSTALL_SPCONV121%"=="" set "INSTALL_SPCONV121=0"
if "%COMPILE_CUDA_EXTENSIONS%"=="" set "COMPILE_CUDA_EXTENSIONS=1"
if "%COMPILE_PCDET_EXTENSIONS%"=="" set "COMPILE_PCDET_EXTENSIONS=0"
if "%SKIP_PYPCD%"=="" set "SKIP_PYPCD=0"

where conda >nul 2>&1
if errorlevel 1 (
    echo [setup][error] Cannot find conda. Run this from an Anaconda/Miniconda prompt.
    exit /b 1
)

echo [setup] Repository: %ROOT_DIR%
echo [setup] Target conda env: %CONDA_ENV_NAME%

conda env list | findstr /r /c:"^[ ]*%CONDA_ENV_NAME%[ ]" >nul 2>&1
if not errorlevel 1 (
    if "%FORCE_RECREATE%"=="1" (
        echo [setup] Removing existing environment: %CONDA_ENV_NAME%
        call conda env remove -n "%CONDA_ENV_NAME%" -y || exit /b 1
    ) else (
        echo [setup] Conda environment already exists. Skipping creation.
        goto activate_env
    )
)

echo [setup] Creating conda environment: %CONDA_ENV_NAME%
call conda create -y -n "%CONDA_ENV_NAME%" python=%PYTHON_VERSION% pip=21.1.2 cmake=3.22.1 || exit /b 1

:activate_env
call conda activate "%CONDA_ENV_NAME%"
if errorlevel 1 (
    echo [setup][error] Failed to activate conda environment: %CONDA_ENV_NAME%
    exit /b 1
)

echo [setup] Python runtime:
python --version || exit /b 1

echo [setup] Installing PyTorch wheels using the OpenCDA Windows convention: %TORCH_VARIANT%
python -m pip install torch==1.10.0+%TORCH_VARIANT% torchvision==0.11.1+%TORCH_VARIANT% torchaudio==0.10.0+%TORCH_VARIANT% -f https://download.pytorch.org/whl/%TORCH_VARIANT%/torch_stable.html --no-deps || exit /b 1
call conda install -y -n "%CONDA_ENV_NAME%" cudnn boost -c conda-forge || exit /b 1

set "REQ_FILE=%TEMP%\opencood_requirements_%RANDOM%.txt"
(
    echo wheel
    echo setuptools^<60
    echo easydict~=1.9
    echo numpy==1.19.5
    echo numba==0.49.0
    echo opencv-python==4.5.5.62
    echo matplotlib==3.3.4
    echo scipy==1.5.4
    echo scikit-image==0.18.3
    echo icecream
    echo tqdm
    echo PyYAML
    echo open3d==0.13.0
    echo cython
    echo tensorboardX
    echo shapely==1.8.5.post1
    echo einops
    echo h5py==3.8.0
    echo pyquaternion
    echo ipdb
    echo python-lzf
) > "%REQ_FILE%"

echo [setup] Installing Python dependencies
python -m pip install --upgrade -r "%REQ_FILE%" || exit /b 1
del "%REQ_FILE%" >nul 2>&1

if "%SKIP_PYPCD%"=="1" (
    echo [setup] Skipping pypcd installation because SKIP_PYPCD=1
) else (
    echo [setup] Installing and patching pypcd for Python 3
    python -m pip install git+https://github.com/klintan/pypcd.git || exit /b 1
    python -c "import pathlib,pypcd; target=pathlib.Path(pypcd.__file__).resolve().parent/'pypcd.py'; text=target.read_text(encoding='utf-8'); text=text.replace('from cStringIO import StringIO','try:\n    from cStringIO import StringIO\nexcept ImportError:\n    from io import StringIO'); text=text.replace('import numpy as np\n','import numpy as np\n\ntry:\n    xrange\nexcept NameError:\n    xrange = range\n') if 'xrange' in text and 'xrange = range' not in text else text; target.write_text(text,encoding='utf-8'); print('patched', target)" || exit /b 1
)

echo [setup] Installing OpenCOOD in editable mode
python -m pip install -e "%ROOT_DIR%" || exit /b 1

if "%COMPILE_CUDA_EXTENSIONS%"=="1" (
    echo [setup] Building OpenCOOD CUDA/Cython bbox extensions
    python "%ROOT_DIR%opencood\utils\setup.py" build_ext --inplace || exit /b 1
) else (
    echo [setup] Skipping OpenCOOD CUDA extension build. Set COMPILE_CUDA_EXTENSIONS=1 to try it.
)

if "%COMPILE_PCDET_EXTENSIONS%"=="1" (
    echo [setup] Building optional OpenCOOD pcdet extensions
    python "%ROOT_DIR%opencood\pcdet_utils\setup.py" build_ext --inplace || exit /b 1
) else (
    echo [setup] Skipping optional pcdet extension build. Set COMPILE_PCDET_EXTENSIONS=1 to try it.
)

if "%INSTALL_SPCONV121%"=="1" (
    where nvcc >nul 2>&1
    if errorlevel 1 (
        echo [setup][error] nvcc is required to build spconv 1.2.1 from source.
        exit /b 1
    )
    set "DEPS_DIR=%ROOT_DIR%.deps"
    set "SPCONV_DIR=%ROOT_DIR%.deps\spconv-v1.2.1"
    if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"
    if not exist "%SPCONV_DIR%" (
        git clone https://github.com/traveller59/spconv.git "%SPCONV_DIR%" || exit /b 1
    )
    pushd "%SPCONV_DIR%" || exit /b 1
    git checkout v1.2.1 || exit /b 1
    git submodule update --init --recursive || exit /b 1
    python setup.py bdist_wheel || exit /b 1
    set "SPCONV_WHEEL="
    for /f "delims=" %%W in ('dir /b /o-d "dist\spconv-1.2.1-*.whl" 2^>nul') do (
        if not defined SPCONV_WHEEL set "SPCONV_WHEEL=%%W"
    )
    if not defined SPCONV_WHEEL (
        echo [setup][error] spconv wheel was not generated.
        exit /b 1
    )
    python -m pip install "dist\!SPCONV_WHEEL!" || exit /b 1
    popd
) else (
    echo [setup] Skipping spconv 1.2.1 build. Set INSTALL_SPCONV121=1 to try it.
)

echo [setup] Running import diagnostics
python -c "import pathlib, sys, torch, opencood; from opencood.version import __version__; print('python', sys.version); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print('opencood', __version__, pathlib.Path(opencood.__file__).resolve())" || exit /b 1
python -c "import pypcd, pathlib; print('pypcd ok', pathlib.Path(pypcd.__file__).resolve())" || echo [setup][warn] pypcd is not importable.
python -c "import spconv, pathlib; print('spconv ok', pathlib.Path(spconv.__file__).resolve())" || echo [setup][warn] spconv is not importable; patched OpenCOOD will use NumPy voxelization fallback.

echo.
echo [setup] Setup completed.
echo [setup] To use:
echo   conda activate %CONDA_ENV_NAME%
echo   cd /d "%ROOT_DIR%"
echo   set PYTHONPATH=%ROOT_DIR%
echo   set PYTHONUTF8=1
echo.
endlocal
