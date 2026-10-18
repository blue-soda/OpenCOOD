"""Six inference-only L1 ordering substitutions on two fixed learned receivers."""
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
from opencood.tools import train_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset

GROUPS=('learned_keep','learned_clear')


def prepare(root,source):
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh probe required')
    final=read(source/'adapt10_summary.json');assert final['passed']
    assert read(source/'round_status.json')['state']=='complete'
    report=dict(source_summary_sha256=sha(source/'adapt10_summary.json'),groups={})
    for group in GROUPS:
        checkpoint=source/(group+'_adapt10/best_ap70.pth')
        assert sha(checkpoint)==final['groups'][group]['checkpoint_sha256']
        entry=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),prefixes={},config_hashes={})
        for prefix in (1,2,3):
            raw=yaml.safe_load((source/group/('p'+str(prefix)+'.yaml')).read_text())
            raw['model']['args']['sender_order']='l1'
            path=root/(group+'_p'+str(prefix)+'.yaml');path.write_text(yaml.safe_dump(raw))
            entry['config_hashes'][str(prefix)]=sha(path)
        cfg=load_yaml(str(root/(group+'_p3.yaml')));data=build_dataset(cfg,train=False)
        batch=train_utils.to_device(data.collate_batch_train([data[0],data[1]]),torch.device('cuda'))['ego']
        model=train_utils.create_model(cfg).cuda().eval();model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
        original_encode,original_membrane=model.encode_lidar,model.encode_membrane
        with torch.no_grad():
            ego,membrane=original_membrane(batch['processed_lidar'],2);road=original_encode(batch['road_lidar'],2)
            model.encode_membrane=lambda lidar,n:(ego,membrane);model.encode_lidar=lambda lidar,n:road
            for prefix in (1,2,3):
                model.readout_prefix=prefix;model.packet_roundtrip=False;direct=model(batch)
                model.packet_roundtrip=True;wire=model(batch)
                assert all(torch.equal(direct[k],wire[k]) for k in ('psm','rm','dm'))
                entry['prefixes'][str(prefix)]=dict(bytes=wire['packet_bytes'],exact_direct_wire=True)
            saved=torch.load(str(checkpoint),map_location='cpu')
            assert all(torch.equal(v.cpu(),saved[k]) for k,v in model.state_dict().items())
        # Break cached closures before dropping GPU model.
        model.encode_lidar=original_encode;model.encode_membrane=original_membrane
        del model,original_encode,original_membrane,ego,membrane,road,batch,data,direct,wire,saved
        gc.collect();torch.cuda.empty_cache();report['groups'][group]=entry
    report['passed']=True;write_json(root/'audit.json',report)
    shutil.copyfile(str(source/'supervise.py'),str(root/'supervise.py'))


def coordinate(root):
    try:
        setup=read(root/'launch.json');audit=read(root/'audit.json');processes=[]
        for gpu in setup['gpus']:free_gpu(gpu)
        for group,gpu in zip(GROUPS,setup['gpus']):
            jobs=[]
            for prefix in (1,2,3):
                name=group+'_p'+str(prefix);folder=root/name
                jobs.append(dict(name=name,timeout_seconds=3600,command=[sys.executable,'-u','-m',
                    'opencood.tools.train_e3dsnn_single','--mode','eval','--workers','2','--seed','20260929',
                    '--config',str(root/(name+'.yaml')),'--checkpoint',audit['groups'][group]['checkpoint'],
                    '--output',str(folder),'--eval-3d','--save-predictions','--packet-roundtrip','--profile']))
                jobs.append(dict(name=name+'_replay',timeout_seconds=3600,command=['env','CUDA_VISIBLE_DEVICES=',sys.executable,
                    '-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(folder)]))
            plan=root/(group+'_plan.json');write_json(plan,dict(name=group,jobs=jobs,gpu=gpu))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (root/(group+'_supervisor.log')).open('xb') as log:
                p=subprocess.Popen([sys.executable,str(root/'supervise.py'),str(plan)],cwd=str(REPO),env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((group,p))
        write_json(root/'status.json',dict(state='running',supervisors={g:p.pid for g,p in processes}))
        while any(p.poll() is None for _,p in processes):
            if any(p.poll() not in (None,0) for _,p in processes):raise RuntimeError('Probe pipeline failed')
            time.sleep(15)
        if any(p.returncode for _,p in processes):raise RuntimeError('Probe pipeline failed')
        final=dict(groups={},source_summary_sha256=audit['source_summary_sha256'])
        for group in GROUPS:
            assert read(root/(group+'_pipeline.json'))['state']=='complete'
            final['groups'][group]={}
            for prefix in (1,2,3):
                item=evaluation(root/(group+'_p'+str(prefix)))
                assert item['manifest']['checkpoint_sha256']==audit['groups'][group]['checkpoint_sha256']
                assert item['manifest']['config_sha256']==audit['groups'][group]['config_hashes'][str(prefix)]
                assert item['manifest']['progressive']['sender_order']=='l1'
                final['groups'][group][str(prefix)]=item
        final['passed']=True;write_json(root/'summary.json',final)
        write_json(root/'status.json',dict(state='complete',finished_unix=time.time()))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry. Other pipelines are not killed.'))
        raise


def launch(root,source,gpus):
    audit=read(root/'audit.json');assert audit['passed']
    assert sha(source/'adapt10_summary.json')==audit['source_summary_sha256']
    if len(set(gpus))!=2:raise ValueError('Two distinct GPUs required')
    if (root/'launch.json').exists():raise FileExistsError('Already launched')
    for group in GROUPS:
        assert sha(audit['groups'][group]['checkpoint'])==audit['groups'][group]['checkpoint_sha256']
        for prefix,digest in audit['groups'][group]['config_hashes'].items():assert sha(root/(group+'_p'+prefix+'.yaml'))==digest
    if shutil.disk_usage(str(root)).free<5*1024**3:raise RuntimeError('Insufficient reserve')
    for gpu in gpus:free_gpu(gpu)
    setup=dict(gpus=gpus,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),audit_sha256=sha(root/'audit.json'),started_unix=time.time())
    write_json(root/'launch.json',setup)
    with (root/'coordinator.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.probe_e3dsnn_progressive_l1','--root',str(root),'--coordinate'],
            cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(coordinator_pid=p.pid,**setup)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--source',default='/data1/tzm/workspace/diagnostics/e3dsnn_progressive_round1_20261018')
    parser.add_argument('--gpus',nargs=2,type=int,default=[1,2]);parser.add_argument('--prepare',action='store_true');parser.add_argument('--coordinate',action='store_true')
    args=parser.parse_args();root=Path(args.root);source=Path(args.source)
    if args.prepare:prepare(root,source)
    elif args.coordinate:coordinate(root)
    else:launch(root,source,args.gpus)
