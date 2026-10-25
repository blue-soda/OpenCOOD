import subprocess,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['cosdh_common_full01','infocom_common_full01_compat']);p.add_argument('--tag',default='replay_watch');a=p.parse_args()
assert a.tag.replace('_','').isalnum()
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');target=r/(a.tag+'_launch.json')
if target.exists():raise RuntimeError('Already launched')
with (r/(a.tag+'.log')).open('wb') as f:
    p=subprocess.Popen(['/data1/zhh/gzc/miniconda3/envs/opencood/bin/python',str(r/'finish_replays.py'),'--tag',a.tag,'--runs']+a.runs,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
target.write_text(json.dumps({'pid':p.pid}));print(p.pid)
