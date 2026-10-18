"""Bounded family x learned/frozen content control; ten epochs, no retries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO,read,sha,free_gpu,evaluation
from opencood.tools.launch_e3dsnn_progressive_round import make_jobs as prefix_jobs,summarize as prefix_summary
from opencood.tools.train_e3dsnn_single import write_json

GROUPS=('snn_fixed','snn_learn','ann_fixed','ann_learn')


def prepare(root,base):
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh content round required')
    report=dict(groups={})
    for group in GROUPS:
        family,treatment=group.split('_');folder=root/group
        command=[sys.executable,'-u','-m','opencood.tools.audit_e3dsnn_content_detector',
                 '--root',str(folder),'--base',str(base),'--family',family]
        if treatment=='learn':command+=['--learn']
        with (root/(group+'_audit.log')).open('xb') as log:
            subprocess.run(command,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        entry=read(folder/'audit.json');assert entry['passed']
        raw=yaml.safe_load((folder/'config.yaml').read_text())
        assert raw['train_params']==dict(batch_size=2,epoches=10) and raw['optimizer']['lr']==5e-5
        entry['config_hashes']={}
        for name in ('p1','p2','p3','off','full'):
            cfg=json.loads(json.dumps(raw))
            if name.startswith('p'):cfg['model']['args']['readout_prefix']=int(name[1:])
            elif name=='off':cfg['model']['args']['fusion_mode']='ego'
            else:cfg['model']['args']['content_full_packet']=True
            path=folder/(name+'.yaml');path.write_text(yaml.safe_dump(cfg));entry['config_hashes'][name]=sha(path)
        report['groups'][group]=entry
    assert len({x['parameters'] for x in report['groups'].values()})==1
    for family in ('snn','ann'):
        a,b=[report['groups'][family+'_'+s] for s in ('fixed','learn')]
        assert a['source_sha256']==b['source_sha256'] and a['initial_basis_sha256']==b['initial_basis_sha256']
        assert a['prefixes']==b['prefixes'] and a['full_packet_bytes']==b['full_packet_bytes']
        assert b['trainable_parameters']-a['trainable_parameters']==8192
        sa,sb=[torch.load(str(root/(family+'_'+s)/'initial.pth'),map_location='cpu') for s in ('fixed','learn')]
        assert set(sa)==set(sb) and all(torch.equal(sa[k],sb[k]) for k in sa)
    report['passed']=True;write_json(root/'audit.json',report)
    shutil.copyfile(str(Path(base)/'e3dsnn_budget_round1_20261018/supervise.py'),str(root/'supervise.py'))


def make_jobs(root,group,stage):
    name,jobs=prefix_jobs(root,group,stage)
    checkpoint=root/group/'initial.pth' if stage=='zero' else root/name/'best_ap70.pth'
    output=root/(name+'_full')
    jobs.extend([dict(name=name+'_full',timeout_seconds=3600,command=[sys.executable,'-u','-m',
        'opencood.tools.train_e3dsnn_single','--workers','2','--seed','20260929','--mode','eval',
        '--config',str(root/group/'full.yaml'),'--checkpoint',str(checkpoint),'--output',str(output),
        '--eval-3d','--save-predictions','--profile','--packet-roundtrip']),
        dict(name=name+'_full_replay',timeout_seconds=3600,command=['env','CUDA_VISIBLE_DEVICES=',sys.executable,
            '-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(output)])])
    return name,jobs


def summarize(root,stage):
    prefix_summary(root,stage);report=read(root/(stage+'_summary.json'));audit=read(root/'audit.json')
    report['passed']=False;write_json(root/(stage+'_summary.json'),report)
    for group,entry in report['groups'].items():
        full=evaluation(root/(group+'_'+stage+'_full'))
        assert full['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
        assert full['manifest']['config_sha256']==audit['groups'][group]['config_hashes']['full']
        # Cached-tensor interface audit requires exact full/event predictions.
        # Independent CUDA passes retain the existing 0.02 percentage-point
        # repeatability bound; each pass must still replay exactly on CPU.
        errors={metric:max(abs(full['evaluation'][metric][k]-entry['prefixes']['p3']['evaluation'][metric][k])
                           for k in ('0.3','0.5','0.7')) for metric in ('ap','ap_3d')}
        assert max(errors.values())<=.0002
        assert full['ordered_gt_sha256']==entry['prefixes']['p3']['ordered_gt_sha256']
        entry['full_event_independent_ap_max_difference']=errors
        entry['prefixes']['full']=full
        if stage=='adapt10':
            state=torch.load(str(root/(group+'_adapt10/best_ap70.pth')),map_location='cpu')
            initial=torch.load(str(root/group/'initial.pth'),map_location='cpu')
            changed=any(not torch.equal(v,state[k]) for k,v in initial.items() if k.startswith('content.'))
            if group.endswith('fixed'):assert not changed
            entry['content_weights_changed']=changed
    if stage=='zero':
        for family in ('snn','ann'):
            a,b=[report['groups'][family+'_'+s]['prefixes'] for s in ('fixed','learn')]
            for prefix in a:
                assert max(abs(a[prefix]['evaluation']['ap'][k]-b[prefix]['evaluation']['ap'][k]) for k in ('0.3','0.5','0.7'))<=.0002
    report['passed']=True;write_json(root/(stage+'_summary.json'),report)


def coordinate(root):
    try:
        setup=read(root/'launch.json')
        for stage in ('zero','adapt10'):
            if stage=='adapt10':assert read(root/'zero_summary.json')['passed']
            for gpu in setup['gpus']:free_gpu(gpu)
            processes=[]
            for group,gpu in zip(GROUPS,setup['gpus']):
                name,jobs=make_jobs(root,group,stage);plan=root/(name+'_plan.json')
                write_json(plan,dict(name=name,jobs=jobs,gpu=gpu))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (root/(name+'_supervisor.log')).open('xb') as log:
                    proc=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((name,proc))
            write_json(root/'round_status.json',dict(state='running',stage=stage,supervisors={n:p.pid for n,p in processes}))
            while any(p.poll() is None for _,p in processes):
                if any(p.poll() not in (None,0) for _,p in processes):raise RuntimeError('Content pipeline failed')
                time.sleep(15)
            if any(p.returncode for _,p in processes):raise RuntimeError('Content pipeline failed')
            summarize(root,stage)
        write_json(root/'round_status.json',dict(state='complete',finished_unix=time.time()))
    except Exception:
        write_json(root/'round_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry; other jobs not killed.'))
        raise


def launch(root,gpus):
    audit=read(root/'audit.json');assert audit['passed']
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    if len(gpus)!=4 or len(set(gpus))!=4:raise ValueError('Four distinct free GPUs required')
    for group in GROUPS:
        assert sha(root/group/'initial.pth')==audit['groups'][group]['initial_sha256']
        for name,digest in audit['groups'][group]['config_hashes'].items():assert sha(root/group/(name+'.yaml'))==digest
    if shutil.disk_usage(str(root)).free<12*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    setup=dict(gpus=gpus,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        audit_sha256=sha(root/'audit.json'),started_unix=time.time(),training_cap_seconds=21600)
    write_json(root/'launch.json',setup)
    with (root/'coordinator.log').open('xb') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_content_round','--root',str(root),
            '--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(coordinator_pid=proc.pid,**setup)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    p.add_argument('--gpus',nargs=4,type=int,default=[0,1,2,5]);p.add_argument('--prepare',action='store_true');p.add_argument('--coordinate',action='store_true')
    a=p.parse_args();root=Path(a.root)
    if a.prepare:prepare(root,a.base)
    elif a.coordinate:coordinate(root)
    else:launch(root,a.gpus)
