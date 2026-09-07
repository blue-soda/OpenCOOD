@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem OpenCOOD/CoBEVFlow setup for Windows.
rem Defaults are chosen for an isolated local test environment named "opencood".
rem Optional environment variables:
rem   CONDA_ENV_NAME=opencood
rem   CLONE_FROM_ENV=opencda to clone a known-good CUDA environment first
rem   PYTHON_VERSION=3.7.11
rem   TORCH_VARIANT=cu113
rem   FORCE_RECREATE=1
rem   INSTALL_SPCONV2=0 to skip the default spconv 2.x wheel installation
rem   SPCONV2_PACKAGE=spconv-cu113
rem   INSTALL_SPCONV121=1 to build legacy spconv 1.2.1 from source
rem   COMPILE_CUDA_EXTENSIONS=0 to skip the default bbox CUDA extension build
rem   COMPILE_PCDET_EXTENSIONS=1 to build optional FPV-RCNN pcdet CUDA extensions
rem   SKIP_PYPCD=1
rem   PIP_DEFAULT_TIMEOUT=1000
rem   PIP_RETRIES=10
rem   INSTALL_CONDA_CUDNN_BOOST=1 to install README's optional conda cudnn/boost packages
rem   PAUSE_ON_EXIT=0 to disable the final pause

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
cd /d "%ROOT_DIR%" || goto setup_failed

if "%CONDA_ENV_NAME%"=="" set "CONDA_ENV_NAME=opencood"
if "%CLONE_FROM_ENV%"=="" set "CLONE_FROM_ENV=opencda"
if "%PYTHON_VERSION%"=="" set "PYTHON_VERSION=3.7.11"
if "%TORCH_VARIANT%"=="" set "TORCH_VARIANT=cu113"
if "%FORCE_RECREATE%"=="" set "FORCE_RECREATE=0"
if "%INSTALL_SPCONV2%"=="" set "INSTALL_SPCONV2=1"
if "%SPCONV2_PACKAGE%"=="" set "SPCONV2_PACKAGE=spconv-cu113"
if "%INSTALL_SPCONV121%"=="" set "INSTALL_SPCONV121=0"
if "%COMPILE_CUDA_EXTENSIONS%"=="" set "COMPILE_CUDA_EXTENSIONS=1"
if "%COMPILE_PCDET_EXTENSIONS%"=="" set "COMPILE_PCDET_EXTENSIONS=0"
if "%SKIP_PYPCD%"=="" set "SKIP_PYPCD=0"
if "%PIP_DEFAULT_TIMEOUT%"=="" set "PIP_DEFAULT_TIMEOUT=1000"
if "%PIP_RETRIES%"=="" set "PIP_RETRIES=10"
if "%INSTALL_CONDA_CUDNN_BOOST%"=="" set "INSTALL_CONDA_CUDNN_BOOST=0"
if "%PAUSE_ON_EXIT%"=="" set "PAUSE_ON_EXIT=1"

where conda >nul 2>&1
if errorlevel 1 (
    echo [setup][error] Cannot find conda. Run this from an Anaconda/Miniconda prompt.
    goto setup_failed
)

echo [setup] Repository: %ROOT_DIR%
echo [setup] Target conda env: %CONDA_ENV_NAME%

conda env list | findstr /r /c:"^[ ]*%CONDA_ENV_NAME%[ ]" >nul 2>&1
if not errorlevel 1 (
    if "%FORCE_RECREATE%"=="1" (
        echo [setup] Removing existing environment: %CONDA_ENV_NAME%
        call conda env remove -n "%CONDA_ENV_NAME%" -y || goto setup_failed
    ) else (
        echo [setup] Conda environment already exists. Skipping creation.
        goto activate_env
    )
)

echo [setup] Creating conda environment: %CONDA_ENV_NAME%
if not "%CLONE_FROM_ENV%"=="" (
    conda env list | findstr /r /c:"^[ ]*%CLONE_FROM_ENV%[ ]" >nul 2>&1
    if not errorlevel 1 (
        echo [setup] Cloning known-good environment: %CLONE_FROM_ENV% -^> %CONDA_ENV_NAME%
        call conda create -y -n "%CONDA_ENV_NAME%" --clone "%CLONE_FROM_ENV%" || goto setup_failed
        goto activate_env
    ) else (
        echo [setup][warn] Clone source env '%CLONE_FROM_ENV%' not found. Creating a fresh environment.
    )
)
call conda create -y -n "%CONDA_ENV_NAME%" python=%PYTHON_VERSION% pip=21.1.2 cmake=3.22.1 || goto setup_failed

:activate_env
call conda activate "%CONDA_ENV_NAME%"
if errorlevel 1 (
    echo [setup][error] Failed to activate conda environment: %CONDA_ENV_NAME%
    goto setup_failed
)

echo [setup] Python runtime:
python --version || goto setup_failed

python -c "import torch; raise SystemExit(0 if torch.__version__ == '1.10.0+%TORCH_VARIANT%' and torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing PyTorch wheels using the OpenCDA Windows convention: %TORCH_VARIANT%
    python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% torch==1.10.0+%TORCH_VARIANT% torchvision==0.11.1+%TORCH_VARIANT% torchaudio==0.10.0+%TORCH_VARIANT% -f https://download.pytorch.org/whl/%TORCH_VARIANT%/torch_stable.html --no-deps || goto setup_failed
) else (
    echo [setup] PyTorch CUDA wheel already satisfies torch==1.10.0+%TORCH_VARIANT%; skipping download.
)
call :prepend_torch_dll_dir || goto setup_failed
if "%INSTALL_CONDA_CUDNN_BOOST%"=="1" (
    echo [setup] Installing optional conda cudnn/boost packages
    call conda install -y -n "%CONDA_ENV_NAME%" cudnn boost -c conda-forge || goto setup_failed
) else (
    echo [setup] Skipping optional conda cudnn/boost packages. Set INSTALL_CONDA_CUDNN_BOOST=1 to install them.
)

set "REQ_FILE=%TEMP%\opencood_requirements_%RANDOM%.txt"
(
    echo wheel
    echo setuptools^<60
    echo easydict~=1.9
    echo numpy^>=1.20,^<1.22
    echo numba==0.49.0
    echo opencv-python==4.5.2.52
    echo matplotlib==3.4.2
    echo scipy==1.6.3
    echo scikit-image
    echo icecream
    echo tqdm
    echo PyYAML
    echo open3d==0.10.0.0
    echo cython
    echo tensorboardX
    echo shapely==1.8.4
    echo einops
    echo h5py==3.8.0
    echo pyquaternion
    echo ipdb
    echo python-lzf
) > "%REQ_FILE%"

echo [setup] Installing Python dependencies
python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% --upgrade -r "%REQ_FILE%" || goto setup_failed
del "%REQ_FILE%" >nul 2>&1

if "%INSTALL_SPCONV2%"=="1" (
    echo [setup] Installing spconv 2.x wheel package: %SPCONV2_PACKAGE%
    python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% "%SPCONV2_PACKAGE%" || goto setup_failed
    python -c "import spconv; import spconv.pytorch as spconv_pt; print('spconv', spconv.__version__, spconv.__file__)" || goto setup_failed
) else (
    echo [setup] Skipping spconv 2.x wheel installation. Set INSTALL_SPCONV2=1 to install it.
)

if "%SKIP_PYPCD%"=="1" (
    echo [setup] Skipping pypcd installation because SKIP_PYPCD=1
) else (
    echo [setup] Installing and patching pypcd for Python 3
    python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% git+https://github.com/klintan/pypcd.git || goto setup_failed
    python -c "import pathlib,pypcd; target=pathlib.Path(pypcd.__file__).resolve().parent/'pypcd.py'; text=target.read_text(encoding='utf-8'); text=text.replace('from cStringIO import StringIO','try:\n    from cStringIO import StringIO\nexcept ImportError:\n    from io import StringIO'); text=text.replace('import numpy as np\n','import numpy as np\n\ntry:\n    xrange\nexcept NameError:\n    xrange = range\n') if 'xrange' in text and 'xrange = range' not in text else text; target.write_text(text,encoding='utf-8'); print('patched', target)" || goto setup_failed
)

echo [setup] Installing OpenCOOD in editable mode
python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% -e "%ROOT_DIR%" --no-deps || goto setup_failed

if "%COMPILE_CUDA_EXTENSIONS%"=="1" (
    echo [setup] Building OpenCOOD CUDA/Cython bbox extensions
    python "%ROOT_DIR%\opencood\utils\setup.py" build_ext --inplace || goto setup_failed
) else (
    echo [setup] Skipping OpenCOOD CUDA extension build. Set COMPILE_CUDA_EXTENSIONS=1 to try it.
)

if "%COMPILE_PCDET_EXTENSIONS%"=="1" (
    call :ensure_cuda_home || goto setup_failed
    echo [setup] Building optional OpenCOOD pcdet extensions
    python "%ROOT_DIR%\opencood\pcdet_utils\setup.py" build_ext --inplace || goto setup_failed
) else (
    echo [setup] Skipping optional pcdet extension build. Set COMPILE_PCDET_EXTENSIONS=1 to try it.
)

if "%INSTALL_SPCONV121%"=="1" (
    call :ensure_cuda_home || goto setup_failed
    set "DEPS_DIR=%ROOT_DIR%\.deps"
    set "SPCONV_DIR=%ROOT_DIR%\.deps\spconv-v1.2.1"
    if not exist "%DEPS_DIR%" mkdir "%DEPS_DIR%"
    if not exist "%SPCONV_DIR%" (
        git clone https://github.com/traveller59/spconv.git "%SPCONV_DIR%" || goto setup_failed
    )
    pushd "%SPCONV_DIR%" || goto setup_failed
    git checkout v1.2.1 || goto setup_failed
    git submodule update --init --recursive || goto setup_failed
    python setup.py bdist_wheel || goto setup_failed
    set "SPCONV_WHEEL="
    for /f "delims=" %%W in ('dir /b /o-d "dist\spconv-1.2.1-*.whl" 2^>nul') do (
        if not defined SPCONV_WHEEL set "SPCONV_WHEEL=%%W"
    )
    if not defined SPCONV_WHEEL (
        echo [setup][error] spconv wheel was not generated.
        goto setup_failed
    )
    python -m pip install --default-timeout %PIP_DEFAULT_TIMEOUT% --retries %PIP_RETRIES% "dist\!SPCONV_WHEEL!" || goto setup_failed
    popd
) else (
    echo [setup] Skipping legacy spconv 1.2.1 source build. Set INSTALL_SPCONV121=1 to try it.
)

echo [setup] Running import diagnostics
python -c "import pathlib, sys, torch, opencood; from opencood.version import __version__; print('python', sys.version); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print('opencood', __version__, pathlib.Path(opencood.__file__).resolve())" || goto setup_failed
python -c "import pypcd, pathlib; print('pypcd ok', pathlib.Path(pypcd.__file__).resolve())" || echo [setup][warn] pypcd is not importable.
python -c "import spconv, pathlib; import spconv.pytorch as spconv_pt; print('spconv ok', spconv.__version__, pathlib.Path(spconv.__file__).resolve())" || echo [setup][warn] spconv is not importable; patched OpenCOOD will use NumPy voxelization fallback.

echo.
echo [setup] Setup completed.
echo [setup] To use:
echo   conda activate %CONDA_ENV_NAME%
echo   cd /d "%ROOT_DIR%"
echo   set PYTHONPATH=%ROOT_DIR%
echo   set PYTHONUTF8=1
echo.
goto setup_success

:prepend_torch_dll_dir
set "TORCH_LIB_DIR="
for /f "delims=" %%P in ('python -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')" 2^>nul') do (
    if not defined TORCH_LIB_DIR set "TORCH_LIB_DIR=%%P"
)
if defined TORCH_LIB_DIR if exist "!TORCH_LIB_DIR!" (
    set "PATH=!TORCH_LIB_DIR!;!PATH!"
    echo [setup] Added PyTorch DLL directory to PATH: !TORCH_LIB_DIR!
)
exit /b 0

:ensure_cuda_home
if not "%CUDA_HOME%"=="" (
    set "CUDA_PATH=%CUDA_HOME%"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    exit /b 0
)
if not "%CUDA_PATH%"=="" (
    set "CUDA_HOME=%CUDA_PATH%"
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    exit /b 0
)
set "NVCC_EXE="
for /f "delims=" %%N in ('where nvcc 2^>nul') do (
    if not defined NVCC_EXE set "NVCC_EXE=%%N"
)
if defined NVCC_EXE (
    for %%D in ("!NVCC_EXE!") do set "NVCC_DIR=%%~dpD"
    for %%D in ("!NVCC_DIR!..") do set "CUDA_HOME=%%~fD"
    set "CUDA_PATH=!CUDA_HOME!"
    set "PATH=!CUDA_HOME!\bin;!PATH!"
    echo [setup] Detected CUDA_HOME: !CUDA_HOME!
    exit /b 0
)
for %%D in (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
) do (
    if not defined CUDA_HOME if exist "%%~D\bin\nvcc.exe" (
        set "CUDA_HOME=%%~D"
        set "CUDA_PATH=%%~D"
        set "PATH=%%~D\bin;!PATH!"
        echo [setup] Detected CUDA_HOME: %%~D
    )
)
if not "%CUDA_HOME%"=="" exit /b 0
echo [setup][error] CUDA_HOME is required for pcdet_utils/spconv compilation, but nvcc was not found.
echo [setup][error] PyTorch CUDA runtime can still work without nvcc; compiling CUDA extensions requires the full CUDA Toolkit.
exit /b 1

:setup_failed
echo.
echo [setup][error] Setup failed. Review the messages above.
set "SETUP_EXIT_CODE=1"
goto setup_exit

:setup_success
set "SETUP_EXIT_CODE=0"
goto setup_exit

:setup_exit
if "%PAUSE_ON_EXIT%"=="1" (
    echo.
    pause
)
endlocal & exit /b %SETUP_EXIT_CODE%
