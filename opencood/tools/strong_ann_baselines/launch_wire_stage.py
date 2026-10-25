"""Launch the audited wire pass from an immutable script snapshot on GPU3."""
import hashlib,json,subprocess,time,shutil
from pathlib import Path
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');source=Path(__file__).resolve().parent
snapshot=r/'wire_stage_20261025';out=r/'infocom_paired_wire_full01'
if snapshot.exists() or out.exists():raise RuntimeError('Snapshot/output already exists; do not duplicate')
samples=[]
for _ in range(5):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
    row=[int(x.strip()) for x in rows[3].split(',')];samples.append(row)
    if row[1]<8000 or row[2]>30:raise RuntimeError('GPU3 admission failed')
    time.sleep(1)
snapshot.mkdir();hashes={}
for name in ('evaluate_infocom.py','infocom_wire.py'):
    p=source/name;shutil.copy2(str(p),str(snapshot/name));hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
cmd=['/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python',str(snapshot/'evaluate_infocom.py'),'--repo','/data1/zhh/gzc/InfoCom','--model-dir','/data1/zhh/gzc/InfoCom/opencood/logs/dairv2x_coalign_base_0526','--data-root','/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure','--split',str(r/'common_val.json'),'--output',str(out),'--gpu','3','--limit','0','--comm-range','1000000','--checkpoint-compat','--wire-codec']
with out.with_suffix('.log').open('wb') as log:
    p=subprocess.Popen(['timeout','--signal=TERM','14400']+cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':p.pid,'command':cmd,'gpu_samples':samples,'snapshot_sha256':hashes,'launched_unix':time.time()}
out.with_suffix('.launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
