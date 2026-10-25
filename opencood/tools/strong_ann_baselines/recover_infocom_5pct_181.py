"""Recover only the uncompleted 5% point after the 146 CUDA outage.

Uses the same checkpoint, data IDs, inference script and codec, with a native
10% cross-host parity gate. Does not repeat completed 1%/2.5% full evaluations.
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

BASE=Path('/data2/tzm/cp_snn_recovery_20261026')
ROOT=BASE/'infocom_5pct_recovery01'
SOURCE=Path(__file__).resolve().parent
EVIDENCE=BASE/'ann_evidence'
PY=str(BASE/'ann_runtime/bin/python')
CHECKPOINT_SHA='7375da04507b4f543666e15aa02897e41e3c32e08bcb3f74b27ad49fb28c803c'


def read(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,indent=2))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate():
    snapshot=ROOT/'snapshot';sys.path.insert(0,str(snapshot))
    import run_infocom_ratio_curve as helpers
    import infocom_wire as codec
    helpers.ROOT=EVIDENCE;helpers.GPU='4'
    try:
        protocol=read(ROOT/'protocol.json')
        for n,digest in protocol['snapshot_sha256'].items():assert sha(snapshot/n)==digest
        assert sha(BASE/'net_epoch_bestval_at7.pth')==CHECKPOINT_SHA
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(BASE/'OpenCOOD'))
        def run(name,ratio,limit):
            gate=helpers.admission();write(ROOT/'status.json',dict(state='running',stage=name,admission=gate))
            output=ROOT/name
            command=[PY,str(snapshot/'evaluate_infocom.py'),'--repo',str(BASE/'InfoCom'),
                '--model-dir',str(BASE),'--data-root','/data1/tzm/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure',
                '--split',str(EVIDENCE/'common_val.json'),'--output',str(output),'--gpu','4',
                '--limit',str(limit),'--comm-range','1000000','--checkpoint-compat','--wire-codec',
                '--mask-ratio',str(ratio)]
            with (ROOT/(name+'.log')).open('x') as log:
                subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,check=True,timeout=14400)
            helpers.verify_wire(output,codec,limit or 1738)
            return output
        smoke=run('native_181_gate',.1,2)
        reference=EVIDENCE/'infocom_paired_wire_full01'
        for row,ref in zip(read(smoke/'predictions_index.json'),read(reference/'predictions_index.json')[:2]):
            assert sha(reference/ref['file'])==ref['sha256']
            with np.load(str(smoke/row['file'])) as a,np.load(str(reference/ref['file'])) as b:
                assert all(a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in ('pred','score','gt'))
            assert row['wire']['messages'][0]['sha256']==ref['wire']['messages'][0]['sha256']
        write(ROOT/'cross_host_parity.json',dict(passed=True,samples=2,
            prediction_and_message_bitwise_equal=True,reference_host='146',current_host='181'))
        output=run('r005',.05,0)
        with (ROOT/'r005_replay.log').open('x') as log:
            subprocess.run([PY,str(snapshot/'replay_common_gt.py'),'--run',str(output),'--root',str(EVIDENCE)],
                env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=3600)
        replay=read(output/'common_gt_replay.json');assert replay['full_split'] and replay['samples']==1738
        write(ROOT/'summary.json',dict(passed=True,mask_ratio=.05,common_gt=replay,
            wire=read(output/'summary.json')['wire_statistics'],audit=read(output/'wire_cpu_audit.json'),
            protocol=protocol,cross_host_parity=read(ROOT/'cross_host_parity.json')))
        write(ROOT/'status.json',dict(state='complete',sha256=sha(ROOT/'summary.json')))
    except Exception:
        write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False));raise


def launch():
    from opencood.tools.strong_ann_baselines import run_infocom_ratio_curve as helpers
    helpers.GPU='4'
    assert sha(BASE/'net_epoch_bestval_at7.pth')==CHECKPOINT_SHA
    if ROOT.exists():raise FileExistsError('Recovery already exists')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:command=p.read_bytes()
        except OSError:continue
        if b'evaluate_infocom.py' in command:raise RuntimeError('Existing InfoCom inference, no duplicate launch')
    gate=helpers.admission();ROOT.mkdir();snapshot=ROOT/'snapshot';snapshot.mkdir();hashes={}
    for name in ('evaluate_infocom.py','infocom_wire.py','run_infocom_ratio_curve.py',
                 'replay_common_gt.py','recover_infocom_5pct_181.py'):
        shutil.copyfile(str(SOURCE/name),str(snapshot/name));hashes[name]=sha(snapshot/name)
    write(ROOT/'protocol.json',dict(snapshot_sha256=hashes,checkpoint_sha256=CHECKPOINT_SHA,
        original_failed_run='/data1/zhh/gzc/diagnostics/ann_baselines_20261025/infocom_ratio_curve_20261026_r1/r005',
        reason='146 CUDA unavailable; only uncompleted 5pct operating point moved',
        mask_ratio=.05,gpu=4,admission=gate,source_split_sha256=sha(EVIDENCE/'common_val.json'),
        completed_1pct_2p5pct_preserved=True,model_or_budget_changes=False))
    with (ROOT/'pipeline.log').open('x') as log:
        child=subprocess.Popen([PY,str(snapshot/'recover_infocom_5pct_181.py'),'--coordinate'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                     PYTHONPATH=str(BASE/'OpenCOOD')))
    record=dict(pid=child.pid,unix=time.time(),gpu=4);write(ROOT/'launch.json',record);print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coordinate',action='store_true');a=p.parse_args()
    if a.coordinate:coordinate()
    else:launch()
