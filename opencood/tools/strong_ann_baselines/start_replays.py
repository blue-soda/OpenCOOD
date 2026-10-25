import subprocess,json
from pathlib import Path
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');target=r/'replay_watch_launch.json'
if target.exists():raise RuntimeError('Already launched')
with (r/'replay_watch.log').open('wb') as f:
    p=subprocess.Popen(['/data1/zhh/gzc/miniconda3/envs/opencood/bin/python',str(r/'finish_replays.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
target.write_text(json.dumps({'pid':p.pid}));print(p.pid)
