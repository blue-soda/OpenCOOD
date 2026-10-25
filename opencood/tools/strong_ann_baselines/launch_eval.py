import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('method',choices=['cosdh','infocom']);p.add_argument('--limit',type=int,default=0);p.add_argument('--tag',required=True);a=p.parse_args()
root=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');repo={'cosdh':'CoSDH','infocom':'InfoCom'}[a.method];gpu={'cosdh':0,'infocom':2}[a.method]
samples=[]
for _ in range(5):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
    row=[int(x.strip()) for x in rows[gpu].split(',')];samples.append(row)
    if row[1]<8000 or row[2]>30:raise RuntimeError('GPU admission failed: '+str(row))
    time.sleep(1)
out=root/(a.method+'_'+a.tag)
if out.exists() or out.with_suffix('.launch.json').exists():raise RuntimeError('Output exists')
model={'cosdh':'dairv2x_cosdh','infocom':'dairv2x_coalign_base_0526'}[a.method]
cmd=['/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python',str(root/('evaluate_'+a.method+'.py')),'--repo','/data1/zhh/gzc/'+repo,'--model-dir','/data1/zhh/gzc/'+repo+'/opencood/logs/'+model,'--data-root','/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure','--split',str(root/'common_val.json'),'--output',str(out),'--gpu',str(gpu),'--limit',str(a.limit)]
with out.with_suffix('.log').open('wb') as log:
    if a.method=='infocom':cmd.append('--checkpoint-compat')
    child=subprocess.Popen(['timeout','--signal=TERM','14400']+cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'command':cmd,'gpu_samples':samples,'launched_unix':time.time(),'timeout_seconds':14400}
out.with_suffix('.launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
