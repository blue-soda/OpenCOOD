"""Bounded CoSDH full wire replay after mandatory independent smoke audit."""
import hashlib,json,subprocess,time,shutil
from pathlib import Path
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');source=Path(__file__).resolve().parent
audit=json.loads((r/'cosdh_wire_smoke02/wire_cpu_audit.json').read_text())
assert audit['all_predictions_bitwise_equal'] and audit['samples']==2 and audit['packet_sha_and_decode_verified']==4
out=r/'cosdh_wire_full01';snapshot=r/'cosdh_wire_snapshot01'
if out.exists() or snapshot.exists() or out.with_suffix('.launch.json').exists():raise RuntimeError('Already launched/prepared')
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:command=p.read_bytes()
    except (FileNotFoundError,PermissionError):continue
    if b'evaluate_cosdh.py' in command and b'--wire-codec' in command:raise RuntimeError('Another CoSDH wire evaluation is active')
samples=[]
for _ in range(5):
    line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()[4]
    row=[int(x.strip()) for x in line.split(',')];samples.append(row)
    if row[1]<8000 or row[2]>30:raise RuntimeError('GPU4 admission failed')
    time.sleep(1)
snapshot.mkdir();hashes={}
for name in ('evaluate_cosdh.py','cosdh_wire.py','infocom_wire.py','verify_cosdh_wire.py'):
    p=source/name;shutil.copy2(str(p),str(snapshot/name));hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
py='/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python'
cmd=[py,str(snapshot/'evaluate_cosdh.py'),'--repo','/data1/zhh/gzc/CoSDH','--model-dir','/data1/zhh/gzc/CoSDH/opencood/logs/dairv2x_cosdh','--data-root','/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure','--split',str(r/'common_val.json'),'--output',str(out),'--gpu','4','--limit','0','--comm-range','1000000','--wire-codec']
with out.with_suffix('.log').open('wb') as log:
    child=subprocess.Popen(['timeout','--signal=TERM','14400']+cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (r/'cosdh_wire_verify.log').open('wb') as log:
    verifier=subprocess.Popen([py,str(snapshot/'verify_cosdh_wire.py'),'--wait'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':child.pid,'verifier_pid':verifier.pid,'command':cmd,'gpu_samples':samples,'snapshot_sha256':hashes,'launched_unix':time.time(),'timeout_seconds':14400}
out.with_suffix('.launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
