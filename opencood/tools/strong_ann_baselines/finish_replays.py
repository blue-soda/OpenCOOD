"""Bounded CPU-only post-evaluation replay; never launches training or retries inference."""
import argparse,json,subprocess,time
from pathlib import Path
root=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');deadline=time.time()+14400
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['cosdh_common_full01','infocom_common_full01_compat']);p.add_argument('--tag',default='replay_watch');a=p.parse_args()
assert a.tag.replace('_','').isalnum()
runs=list(a.runs);state={}
while runs and time.time()<deadline:
    for name in list(runs):
        run=root/name
        if (run/'failure.json').exists():
            state[name]={'status':'inference_failed'};runs.remove(name)
        elif (run/'summary.json').exists():
            with (root/(name+'.replay.log')).open('wb') as log:
                code=subprocess.call(['/data1/zhh/gzc/miniconda3/envs/opencood/bin/python',str(root/'replay_common_gt.py'),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT)
            state[name]={'status':'replayed' if code==0 else 'replay_failed','returncode':code};runs.remove(name)
    (root/(a.tag+'_status.json')).write_text(json.dumps({'completed':state,'pending':runs,'updated_unix':time.time()},indent=2))
    if runs:time.sleep(30)
if runs:
    (root/(a.tag+'_status.json')).write_text(json.dumps({'completed':state,'timed_out':runs},indent=2))
