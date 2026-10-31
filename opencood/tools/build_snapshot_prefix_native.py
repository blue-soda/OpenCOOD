"""Build the opt-in CPU codec accelerator in an explicitly named private path."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from opencood.utils.snapshot_prefix_native import library

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    output = Path(a.output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / 'utils/snapshot_prefix_native.cpp'
    command = ['g++', '-O3', '-std=c++11', '-shared', '-fPIC', '-fopenmp', str(source), '-lz', '-o', str(output)]
    subprocess.run(command, check=True)
    library(str(output))
    report = dict(command=command, library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(['g++', '--version']).decode().splitlines()[0])
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
