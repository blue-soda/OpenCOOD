"""Verify completed fixed2k artifacts and launch only final evaluation on 181."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import torch
import yaml


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2))


def migrate(a):
    root=Path(a.root).resolve();archive=Path(a.archive).resolve()
    assert sha(archive)==a.archive_sha256
    assert read(root/'standby.json')['state']=='standby_only'
    if (root/'migration.json').exists():raise FileExistsError('Migration already prepared; inspect it before any retry')
    imported=root/'original_146';imported.mkdir()
    with tarfile.open(str(archive)) as tf:
        for member in tf.getmembers():
            rel=Path(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            target=imported/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    hashes=read(imported/'fixed2k_completed_sha256.json')
    assert all(sha(imported/n)==h for n,h in hashes.items())
    old=imported/'fixed2k';new=root/'fixed2k'
    assert read(old/'status.json')['state']=='failed'
    assert 'CUDA required' in (old/'b0.log').read_text()
    assert read(old/'train/status.json')['state']=='complete'
    assert read(old/'audit.json')['passed']
    frozen=read(old/'train/fixed_state.json');assert frozen['passed'] and frozen['verified_through_epoch']==10
    assert read(old/'train/fixed_bn.json')['passed']
    state=torch.load(str(old/'train/last.pt'),map_location='cpu');assert state['epoch']==10
    assert state['config_sha256']==sha(old/'train.yaml')
    original=read(imported/'protocol.json');rebased=read(root/'protocol.json')
    assert sha(root/'source.pth')==original['source_sha256']==rebased['source_sha256']
    checked=[]
    for name in ('train','b0','b2048','b4096','b8192'):
        rel='fixed2k/'+name+'.yaml'
        assert sha(imported/rel)==original['config_hashes'][rel]
        assert sha(root/rel)==rebased['config_hashes'][rel]
        before=yaml.safe_load((imported/rel).read_text());after=yaml.safe_load((root/rel).read_text())
        normalized=copy.deepcopy(before)
        for key in ('data_dir','root_dir','validate_dir','test_dir','data_manifest'):normalized[key]=after[key]
        if name!='train':normalized['model']['args']['snapshot']['packet_directory']=after['model']['args']['snapshot']['packet_directory']
        assert normalized==after,'Unexpected semantic config difference: '+rel
        checked.append(rel)
    cfg=yaml.safe_load((new/'train.yaml').read_text())
    manifest=read(Path(cfg['data_manifest']))
    for split,key in (('train','root_dir'),('val','validate_dir')):
        assert sha(Path(cfg[key]))==manifest['splits'][split]['source_sha256']==original['dataset_hashes'][key]
    assert manifest['splits']['train']['valid_count']==4678 and manifest['splits']['val']['valid_count']==1738
    assert sha(Path(cfg['data_dir'])/'cooperative/data_info.json')==manifest['cooperative_info_sha256']
    assert state['data_manifest_sha256']==original['dataset_hashes']['data_manifest']
    del state
    (new/'train').mkdir()
    for p in (old/'train').iterdir():
        # Same filesystem hard links retain exact immutable completed artifacts.
        os.link(str(p),str(new/'train'/p.name))
    shutil.copyfile(str(old/'audit.json'),str(new/'audit.json'))
    record=dict(passed=True,archive_sha256=a.archive_sha256,checked_configs=checked,
        host_changes_only=True,training_complete_epoch=10,training_restarted=False,
        original_failure_preserved=True,best_sha256=sha(new/'train/best_ap70.pth'),
        original_selected_metrics=read(new/'train/best_ap70_metrics.json'))
    write(root/'migration.json',record)
    free=shutil.disk_usage(str(root)).free
    if free<1024**3:raise RuntimeError('Less than 1GiB free for final evaluation')
    readings=[]
    for _ in range(5):
        free_mib,util=map(int,subprocess.check_output(['nvidia-smi','-i',a.gpu,
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().split(','))
        if free_mib<16000 or util>20:raise RuntimeError('GPU admission failed')
        readings.append([free_mib,util]);time.sleep(1)
    repo=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-u','-m','opencood.tools.run_e3dsnn_snapshot_joint',
        '--root',str(root),'--arm','fixed2k','--mode','finish']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(repo),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCOOD_CUDA_MEMORY_FRACTION='.6')
    with (root/'finish.log').open('x') as log:
        process=subprocess.Popen(command,cwd=str(repo),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=process.pid,command=command,admission=readings,unix=time.time(),
        gpu=a.gpu,free_disk_bytes=free,training_restarted=False)
    write(root/'finish_launch.json',launch);print(json.dumps(launch))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','archive','archive-sha256','gpu'):p.add_argument('--'+n,required=True)
    migrate(p.parse_args())
