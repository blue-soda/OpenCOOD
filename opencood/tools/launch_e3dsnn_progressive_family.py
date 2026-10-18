"""Two matched-budget families using fixed L1 progressive message order."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO,read,sha,free_gpu
from opencood.tools.launch_e3dsnn_progressive_round import make_jobs,summarize
from opencood.tools.train_e3dsnn_single import write_json

FAMILIES=('snn','ann')


def prepare(root,base):
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh family round required')
    report=dict(groups={})
    for family in FAMILIES:
        folder=root/family
        with (root/(family+'_audit.log')).open('xb') as log:
            subprocess.run([sys.executable,'-u','-m','opencood.tools.audit_e3dsnn_progressive_detector',
                '--root',str(folder),'--base',str(base),'--family',family],cwd=str(REPO),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        entry=read(folder/'audit.json');assert entry['passed']
        raw=yaml.safe_load((folder/'config.yaml').read_text())
        assert raw['train_params']==dict(batch_size=2,epoches=10) and raw['optimizer']['lr']==5e-5
        entry['config_hashes']={}
        for prefix in (1,2,3,0):
            cfg=json.loads(json.dumps(raw))
            if prefix:cfg['model']['args']['readout_prefix']=prefix
            else:cfg['model']['args']['fusion_mode']='ego'
            name='p'+str(prefix) if prefix else 'off';path=folder/(name+'.yaml')
            path.write_text(yaml.safe_dump(cfg));entry['config_hashes'][name]=sha(path)
        entry['budget_codec']=raw['model']['args']['budget_codec'];report['groups'][family]=entry
    assert len({v['parameters'] for v in report['groups'].values()})==1
    assert len({v['trainable_parameters'] for v in report['groups'].values()})==1
    assert report['groups']['snn']['budget_codec']==dict(bits=3,keep_fraction=.25)
    assert report['groups']['ann']['budget_codec']==dict(bits=2,keep_fraction=.5)
    report['passed']=True;write_json(root/'audit.json',report)
    shutil.copyfile(str(Path(base)/'e3dsnn_budget_round1_20261018/supervise.py'),str(root/'supervise.py'))


def coordinate(root):
    try:
        setup=read(root/'launch.json')
        for stage in ('zero','adapt10'):
            if stage=='adapt10':assert read(root/'zero_summary.json')['passed']
            for gpu in setup['gpus']:free_gpu(gpu)
            processes=[]
            for family,gpu in zip(FAMILIES,setup['gpus']):
                name,jobs=make_jobs(root,family,stage);plan=root/(name+'_plan.json')
                write_json(plan,dict(name=name,jobs=jobs,gpu=gpu))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (root/(name+'_supervisor.log')).open('xb') as log:
                    p=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((name,p))
            write_json(root/'round_status.json',dict(state='running',stage=stage,supervisors={n:p.pid for n,p in processes}))
            while any(p.poll() is None for _,p in processes):
                if any(p.poll() not in (None,0) for _,p in processes):raise RuntimeError('Family pipeline failed')
                time.sleep(15)
            if any(p.returncode for _,p in processes):raise RuntimeError('Family pipeline failed')
            summarize(root,stage)
        write_json(root/'round_status.json',dict(state='complete',finished_unix=time.time()))
    except Exception:
        write_json(root/'round_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry; other pipelines not killed.'))
        raise


def launch(root,gpus):
    audit=read(root/'audit.json');assert audit['passed']
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    if len(set(gpus))!=2:raise ValueError('Two distinct GPUs required')
    for family in FAMILIES:
        assert sha(root/family/'initial.pth')==audit['groups'][family]['initial_sha256']
        for name,digest in audit['groups'][family]['config_hashes'].items():assert sha(root/family/(name+'.yaml'))==digest
    if shutil.disk_usage(str(root)).free<8*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    setup=dict(gpus=gpus,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        audit_sha256=sha(root/'audit.json'),started_unix=time.time(),training_cap_seconds=21600)
    write_json(root/'launch.json',setup)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_progressive_family',
            '--root',str(root),'--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(coordinator_pid=p.pid,**setup)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics');parser.add_argument('--gpus',nargs=2,type=int,default=[1,2])
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--coordinate',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    if args.prepare:prepare(root,args.base)
    elif args.coordinate:coordinate(root)
    else:launch(root,args.gpus)
