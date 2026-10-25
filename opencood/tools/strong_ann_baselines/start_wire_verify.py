import subprocess,json
from pathlib import Path
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');source=Path(__file__).resolve().parent;target=r/'wire_full_verify_launch.json'
if target.exists():raise RuntimeError('Already started')
with (r/'wire_full_verify.log').open('wb') as log:
    p=subprocess.Popen(['/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python',str(source/'verify_wire_full.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
target.write_text(json.dumps({'pid':p.pid}));print(p.pid)
