"""Bounded order-learning x residual-retention round; full-prefix AP70 selection."""
import argparse
import gc
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
from opencood.tools.train_e3dsnn_single import write_json

GROUPS=('frozen_keep','frozen_clear','learned_keep','learned_clear')


def prepare(root,base):
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh round required')
    report=dict(groups={})
    for group in GROUPS:
        order,residual=group.split('_');folder=root/group
        command=[sys.executable,'-u','-m','opencood.tools.audit_e3dsnn_progressive_detector',
                 '--root',str(folder),'--base',str(base),'--sender-order',order]
        if residual=='clear':command+=['--clear-residual']
        with (root/(group+'_audit.log')).open('xb') as log:
            subprocess.run(command,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        entry=read(folder/'audit.json');assert entry['passed']
        raw=yaml.safe_load((folder/'config.yaml').read_text())
        assert raw['train_params']==dict(batch_size=2,epoches=10)
        assert raw['optimizer']['lr']==5e-5
        entry['config_hashes']={}
        for prefix in (1,2,3,0):
            cfg=json.loads(json.dumps(raw))
            if prefix:cfg['model']['args']['readout_prefix']=prefix
            else:cfg['model']['args']['fusion_mode']='ego'
            name='p'+str(prefix) if prefix else 'off'
            path=folder/(name+'.yaml');path.write_text(yaml.safe_dump(cfg))
            entry['config_hashes'][name]=sha(path)
        report['groups'][group]=entry
    assert len({x['initial_sha256'] for x in report['groups'].values()})==1
    assert len({x['encoder_sha256'] for x in report['groups'].values()})==1
    assert len({x['decoder_sha256'] for x in report['groups'].values()})==1
    for residual in ('keep','clear'):
        assert report['groups']['frozen_'+residual]['prefixes']==report['groups']['learned_'+residual]['prefixes']
    report['passed']=True;write_json(root/'audit.json',report)
    shutil.copyfile(str(Path(base)/'e3dsnn_budget_round1_20261018/supervise.py'),str(root/'supervise.py'))


def make_jobs(root,group,stage):
    name=group+'_'+stage;folder=root/group
    base=[sys.executable,'-u','-m','opencood.tools.train_e3dsnn_single','--workers','2','--seed','20260929']
    checkpoint=folder/'initial.pth' if stage=='zero' else root/name/'best_ap70.pth'
    jobs=[]
    if stage=='adapt10':
        jobs.append(dict(name=name+'_train',timeout_seconds=21600,command=base+['--mode','train',
            '--config',str(folder/'p3.yaml'),'--checkpoint',str(folder/'initial.pth'),
            '--output',str(root/name),'--epochs','10','--selection-iou','0.7','--retain-initial-best']))
    for mode in (('p1','p2','p3') if stage=='zero' else ('p1','p2','p3','off')):
        output=root/(name+'_'+mode)
        command=base+['--mode','eval','--config',str(folder/(mode+'.yaml')),
            '--checkpoint',str(checkpoint),'--output',str(output),'--eval-3d','--save-predictions','--profile']
        if mode!='off':command+=['--packet-roundtrip']
        jobs.append(dict(name=name+'_'+mode,timeout_seconds=3600,command=command))
        jobs.append(dict(name=name+'_'+mode+'_replay',timeout_seconds=3600,command=[
            'env','CUDA_VISIBLE_DEVICES=',sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions',
            '--evaluation-dir',str(output)]))
    return name,jobs


def summarize(root,stage):
    audit=read(root/'audit.json');report=dict(stage=stage,groups={})
    for group in GROUPS:
        name=group+'_'+stage;assert read(root/(name+'_pipeline.json'))['state']=='complete'
        checkpoint=root/group/'initial.pth' if stage=='zero' else root/name/'best_ap70.pth'
        entry=dict(checkpoint_sha256=sha(checkpoint),prefixes={})
        for mode in (('p1','p2','p3') if stage=='zero' else ('p1','p2','p3','off')):
            item=evaluation(root/(name+'_'+mode))
            assert item['manifest']['checkpoint_sha256']==entry['checkpoint_sha256']
            assert item['manifest']['config_sha256']==audit['groups'][group]['config_hashes'][mode]
            if mode!='off':assert item['evaluation']['feature_packets']['count']==1738
            entry['prefixes'][mode]=item
        rows=[entry['prefixes'][p]['communication_frames'] for p in ('p1','p2','p3')]
        assert all(a['frame_id']==b['frame_id']==c['frame_id'] and a['bytes']<=b['bytes']<=c['bytes'] for a,b,c in zip(*rows))
        if stage=='adapt10':
            for filename,key in (('frozen_encoder.json','encoder_sha256'),('frozen_decoder.json','decoder_sha256'),('fixed_bn.json','bn_sha256')):
                item=read(root/name/filename)
                assert item['passed'] and item['verified_through_epoch']==10
                assert item['initial_sha256']==item['current_sha256']==audit['groups'][group][key]
                entry[filename]=item
            entry['best_metrics']=read(root/name/'best_ap70_metrics.json')
            state=torch.load(str(checkpoint),map_location='cpu')
            initial=torch.load(str(root/group/'initial.pth'),map_location='cpu')
            changed=any(not torch.equal(state[k],v) for k,v in initial.items() if k.startswith('progressive_sender.'))
            if group.startswith('frozen'):assert not changed
            entry['selected_sender_weights_changed']=changed
            del state,initial;gc.collect()
        report['groups'][group]=entry
    if stage=='zero':
        for residual in ('keep','clear'):
            for prefix in ('p1','p2','p3'):
                a=report['groups']['frozen_'+residual]['prefixes'][prefix]
                b=report['groups']['learned_'+residual]['prefixes'][prefix]
                assert max(abs(a['evaluation']['ap'][k]-b['evaluation']['ap'][k]) for k in ('0.3','0.5','0.7'))<=.0002
    report['passed']=True;write_json(root/(stage+'_summary.json'),report)


def coordinate(root):
    setup=read(root/'launch.json')
    try:
        for stage in ('zero','adapt10'):
            if stage=='adapt10':assert read(root/'zero_summary.json')['passed']
            for gpu in setup['gpus']:free_gpu(gpu)
            processes=[]
            for group,gpu in zip(GROUPS,setup['gpus']):
                name,jobs=make_jobs(root,group,stage);plan=root/(name+'_plan.json')
                write_json(plan,dict(name=name,jobs=jobs,gpu=gpu))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (root/(name+'_supervisor.log')).open('xb') as log:
                    p=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append((name,p))
            write_json(root/'round_status.json',dict(state='running',stage=stage,supervisors={n:p.pid for n,p in processes}))
            while any(p.poll() is None for _,p in processes):
                if any(p.poll() not in (None,0) for _,p in processes):raise RuntimeError('Pipeline failed')
                time.sleep(15)
            if any(p.returncode for _,p in processes):raise RuntimeError('Pipeline failed')
            summarize(root,stage)
        write_json(root/'round_status.json',dict(state='complete',finished_unix=time.time()))
    except Exception:
        write_json(root/'round_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry; other pipelines are not killed.'))
        raise


def launch(root,gpus):
    audit=read(root/'audit.json');assert audit['passed']
    if (root/'launch.json').exists():raise FileExistsError('Already launched')
    if len(set(gpus))!=4:raise ValueError('Four distinct GPUs required')
    for group in GROUPS:
        assert sha(root/group/'initial.pth')==audit['groups'][group]['initial_sha256']
        for name,digest in audit['groups'][group]['config_hashes'].items():assert sha(root/group/(name+'.yaml'))==digest
    if shutil.disk_usage(str(root)).free<12*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    setup=dict(gpus=gpus,git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        audit_sha256=sha(root/'audit.json'),started_unix=time.time(),training_cap_seconds=21600)
    write_json(root/'launch.json',setup)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_progressive_round',
            '--root',str(root),'--coordinate'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(coordinator_pid=p.pid,**setup)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics');parser.add_argument('--gpus',nargs=4,type=int,default=[1,2,3,4])
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--coordinate',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    if args.prepare:prepare(root,args.base)
    elif args.coordinate:coordinate(root)
    else:launch(root,args.gpus)
