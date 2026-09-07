import os
from pathlib import Path


def _prepend_torch_dll_dir():
    """Make PyTorch-bundled CUDA DLLs visible to extension wheels on Windows."""
    if os.name != "nt":
        return
    try:
        import torch
    except Exception:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.is_dir():
        return

    torch_lib_str = str(torch_lib)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if torch_lib_str not in path_parts:
        os.environ["PATH"] = torch_lib_str + os.pathsep + os.environ.get("PATH", "")

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(torch_lib_str)
        except OSError:
            pass


_prepend_torch_dll_dir()
