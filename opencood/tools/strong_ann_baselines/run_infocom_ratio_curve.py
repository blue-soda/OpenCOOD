"""Three preregistered frozen InfoCom mask-ratio points; same GPU, actual wire.

Starts with a 10% equivalence gate against saved native predictions. Lower
ratios are inference probes, not fully adapted low-budget ANN baselines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025')
SOURCE=Path(__file__).resolve().parent
PY='/data1/zhh/gzc/miniconda3/envs/ann_runtime/bin/python'
GPU='4'
POINTS=(('r001',.01),('r0025',.025),('r005',.05))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,indent=2))


def admission():
    # A just-finished own smoke can leave the utilization sampling window high
    # after its CUDA context has gone. Let that window expire, then require all
    # five fresh readings to pass; do not relax occupancy or auto-retry gates.
    time.sleep(3)
    readings=[]
    for _ in range(5):
        line=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free,utilization.gpu',
            '--format=csv,noheader,nounits']).decode().strip()
        free,util=map(int,line.split(','));readings.append([free,util])
        if free<8000 or util>20:raise RuntimeError('GPU4 admission failed; no automatic retry: '+line)
        time.sleep(1)
    return readings


def verify_wire(run,codec,limit):
    index=read(run/'predictions_index.json');ids=read(ROOT/'common_val.json')[:limit]
    assert len(index)==len(ids)==limit
    for row,fid in zip(index,ids):
        assert row['frame_id']==fid and row['record_len']==[2]
        assert sha(run/row['file'])==row['sha256']
        assert len(row['wire']['messages'])==1
        total=0
        for message in row['wire']['messages']:
            p=run/message['file'];blob=p.read_bytes();assert sha(p)==message['sha256']
            frame,sender,seq,*pose=codec.ENVELOPE.unpack_from(blob)
            assert frame==int(fid) and sender==1 and seq==0 and np.isfinite(pose).all()
            latent,mask=codec.decode(blob[codec.ENVELOPE.size:])
            assert latent.shape==(256,) and mask.shape==(200,504)
            assert np.isfinite(latent).all() and np.isfinite(mask).all()
            total+=len(blob)
        assert total==row['wire']['total_bytes']
    write(run/'wire_cpu_audit.json',dict(passed=True,samples=limit,all_packet_and_prediction_sha_verified=True,
        envelope_and_shapes_verified=True,source_index_sha256=sha(run/'predictions_index.json')))


def coordinate(root):
    snapshot=root/'snapshot';sys.path.insert(0,str(snapshot))
    import infocom_wire as codec
    try:
        protocol=read(root/'protocol.json')
        for name,digest in protocol['snapshot_sha256'].items():assert sha(snapshot/name)==digest
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        def run(name,ratio,limit):
            out=root/name
            gate=admission();write(root/'status.json',dict(state='running',stage=name,admission=gate))
            cmd=[PY,str(snapshot/'evaluate_infocom.py'),'--repo','/data1/zhh/gzc/InfoCom',
                '--model-dir','/data1/zhh/gzc/InfoCom/opencood/logs/dairv2x_coalign_base_0526',
                '--data-root','/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure',
                '--split',str(ROOT/'common_val.json'),'--output',str(out),'--gpu',GPU,
                '--limit',str(limit),'--comm-range','1000000','--checkpoint-compat','--wire-codec',
                '--mask-ratio',str(ratio)]
            with (root/(name+'.log')).open('x') as log:
                subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,check=True,timeout=14400)
            verify_wire(out,codec,limit or 1738)
            return out
        smoke=run('native_ratio_gate',.1,2)
        original=ROOT/'infocom_paired_wire_full01'
        for row,ref in zip(read(smoke/'predictions_index.json'),read(original/'predictions_index.json')[:2]):
            with np.load(str(smoke/row['file'])) as a,np.load(str(original/ref['file'])) as b:
                assert all(a[k].tobytes()==b[k].tobytes() and a[k].shape==b[k].shape for k in ('pred','score','gt'))
            assert row['wire']['total_bytes']==ref['wire']['total_bytes']
            assert row['wire']['messages'][0]['sha256']==ref['wire']['messages'][0]['sha256']
        write(root/'native_equivalence.json',dict(passed=True,samples=2,predictions_and_wire_bitwise_equal=True))
        results={}
        for name,ratio in POINTS:
            out=run(name,ratio,0)
            with (root/(name+'_replay.log')).open('x') as log:
                subprocess.run([PY,str(snapshot/'replay_common_gt.py'),'--run',str(out),'--root',str(ROOT)],
                    env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,check=True,timeout=3600)
            replay=read(out/'common_gt_replay.json');assert replay['full_split'] and replay['samples']==1738
            results[name]=dict(mask_ratio=ratio,common_gt=replay,
                wire=read(out/'summary.json')['wire_statistics'],audit=read(out/'wire_cpu_audit.json'))
            # Intermediate results are retained but no AP-dependent additions.
            write(root/'results_so_far.json',results)
        write(root/'summary.json',dict(passed=True,points=results,protocol=protocol,
            limitation='Frozen checkpoint operating curve, not matched adaptation or SNN superiority evidence'))
        write(root/'status.json',dict(state='complete',sha256=sha(root/'summary.json')))
    except Exception:
        write(root/'status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False));raise


def launch(root):
    if root.exists():raise FileExistsError('Refuse duplicate or partial run')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=path.read_bytes()
        except (OSError,PermissionError):continue
        if b'evaluate_infocom.py' in command:raise RuntimeError('Another InfoCom evaluation is active')
    gate=admission();root.mkdir();snapshot=root/'snapshot';snapshot.mkdir();hashes={}
    for name in ('evaluate_infocom.py','infocom_wire.py','replay_common_gt.py','run_infocom_ratio_curve.py'):
        shutil.copyfile(str(SOURCE/name),str(snapshot/name));hashes[name]=sha(snapshot/name)
    write(root/'protocol.json',dict(points=POINTS,gpu=GPU,seed=303,snapshot_sha256=hashes,
        split_sha256=sha(ROOT/'common_val.json'),purpose='Frozen strong-ANN inference rate/AP curve',
        no_hard_byte_cap=True,no_new_training=True,no_ap_dependent_parameter_search=True,
        original_ratio=.1,known_original_ap70=.6457136214761451,known_original_mean_bytes=8754.928078250863,
        admission=gate,timing='Shared GPU operational time only; no compute superiority claim'))
    with (root/'pipeline.log').open('x') as log:
        child=subprocess.Popen([PY,str(snapshot/'run_infocom_ratio_curve.py'),'--root',str(root),'--coordinate'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    record=dict(pid=child.pid,gpu=GPU,unix=time.time(),snapshot_sha256=hashes)
    write(root/'launch.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--coordinate',action='store_true')
    a=p.parse_args()
    if a.coordinate:coordinate(a.root)
    else:launch(a.root)
