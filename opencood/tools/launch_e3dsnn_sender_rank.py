"""Six fixed-weight evaluations of sender-local selection; zero training."""
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
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, read, sha, free_gpu, evaluation
from opencood.tools.train_e3dsnn_single import write_json
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import quantized_feature_packet as codec

FAMILIES = ('snn_ann', 'snn_snn', 'ann_ann')  # encoder, fusion; all ANN BEV/heads
POLICIES = ('l1', 'confidence')


def sources(base):
    base=Path(base);result={}
    assert read(base/'e3dsnn_message_fusion_round1_20261018/round_status.json')['state']=='complete'
    for family, folder, group, prefix, fusion in [
        ('snn_ann','e3dsnn_budget_round1_20261018','s_fusion','budget','relu'),
        ('snn_snn','e3dsnn_message_fusion_round1_20261018','snn_geometry','message','count4'),
        ('ann_ann','e3dsnn_budget_round1_20261018','a_fusion','budget','relu')]:
        root=base/folder
        assert read(root/'round_status.json')['state']=='complete'
        report=read(root/'adapt10_summary.json');assert report['passed']
        entry=report['groups'][group];checkpoint=root/(prefix+'_'+group+'_adapt10/best_ap70.pth')
        assert sha(checkpoint)==entry['checkpoint_sha256']
        config=root/(group+'.yaml');assert sha(config)==read(root/'audit.json')['groups'][group]['config_sha256']['']
        result[family]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            config=str(config),config_sha256=sha(config),source_summary_sha256=sha(root/'adapt10_summary.json'),
            fusion_activation=fusion,reference_ap=entry['on']['evaluation']['ap'],
            reference_bytes=entry['on']['evaluation']['feature_packets'],adaptation_epochs=50)
    return result


def audit(root,base):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'audit.json').exists():raise FileExistsError('Already audited')
    source=sources(base);report=dict(sources=source,families={})
    for family in FAMILIES:
        raw=yaml.safe_load(Path(source[family]['config']).read_text());configs={}
        for policy in POLICIES:
            cfg=json.loads(json.dumps(raw));cfg['name']='rank_'+family+'_'+policy
            cfg['model']['core_method']='e3dsnn_sender_rank'
            cfg['model']['args'].update(sender_ranking=policy,fusion_activation_kind=source[family]['fusion_activation'])
            p=root/(family+'_'+policy+'.yaml')
            with p.open('x') as f:yaml.safe_dump(cfg,f)
            configs[policy]=dict(path=str(p),sha256=sha(p))
        cfg=load_yaml(source[family]['config']);dataset=build_dataset(cfg,train=False)
        native=train_utils.create_model(cfg).cuda().eval()
        native.load_state_dict(torch.load(source[family]['checkpoint'],map_location='cpu'),strict=True)
        model=train_utils.create_model(load_yaml(configs['l1']['path'])).cuda().eval()
        model.load_state_dict(torch.load(source[family]['checkpoint'],map_location='cpu'),strict=True)
        assert not any(p.requires_grad for p in model.parameters())
        original=native.encode_lidar;process=native.process_road_features;frames=[]
        for index in (0,1):
            data=train_utils.to_device(dataset.collate_batch_test([dataset[index]]),torch.device('cuda'))['ego']
            with torch.no_grad():
                cached={id(data[k]):original(data[k],1) for k in ('processed_lidar','road_lidar')}
                native.encode_lidar=lambda lidar,n:cached[id(lidar)];model.encode_lidar=lambda lidar,n:cached[id(lidar)]
                native.packet_roundtrip=True;native.process_road_features=process;reference=native(data)
                road=cached[id(data['road_lidar'])];c=road.indices[:,1:].cpu().numpy();f=road.features.detach().cpu().numpy()
                frame=dict(frame_id=data['frame_ids'][0],policies={});selections={}
                for policy in POLICIES:
                    model.sender_ranking=policy;model.packet_roundtrip=True;output=model(data)
                    assert all(torch.isfinite(output[k]).all() for k in ('psm','rm','dm'))
                    if policy=='l1':
                        assert all(torch.equal(output[k],reference[k]) for k in ('psm','rm','dm'))
                        assert output['packet_bytes']==reference['packet_bytes']
                    scores=model.packet_row_scores(road)
                    coords,codes,scales=codec.prepare(c,f,model.activation,model.budget_codec,scores)
                    dense=torch.zeros_like(road.dense());xyz=torch.as_tensor(coords.astype('int64'),device='cuda')
                    dense[0,:,xyz[:,0],xyz[:,1],xyz[:,2]]=torch.as_tensor((codes.astype(np.float32)*scales).T.copy(),device='cuda')
                    cached[id(data['road_lidar'])]=SimpleNamespace(dense=lambda:dense)
                    native.process_road_features=lambda sparse:sparse;native.packet_roundtrip=False;oracle=native(data)
                    cached[id(data['road_lidar'])]=road
                    assert all(torch.equal(output[k],oracle[k]) for k in ('psm','rm','dm'))
                    frame['policies'][policy]=dict(packet_bytes=output['packet_bytes'][0],retained_rows=len(coords),oracle_exact=True)
                    selections[policy]={tuple(x) for x in coords.tolist()}
                frame['selection_intersection']=len(selections['l1'] & selections['confidence'])
                frames.append(frame)
        report['families'][family]=dict(configs=configs,frames=frames)
        del model,native,original,process,dataset,data,cached,road,reference,output,oracle,dense
        gc.collect();torch.cuda.empty_cache()
    report['passed']=True
    with (root/'audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


def commands(root,family,policy,source,config):
    out=root/(family+'_'+policy)
    return [([sys.executable,'-u','-m','opencood.tools.train_e3dsnn_single','--mode','eval',
        '--config',config,'--checkpoint',source['checkpoint'],'--output',str(out),'--workers','2',
        '--seed','20260929','--eval-3d','--save-predictions','--packet-roundtrip','--profile'],False),
        ([sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(out)],True)]


def run_family(root,family):
    report=read(root/'audit.json');assert report['passed'];source=report['sources'][family]
    assert sha(source['checkpoint'])==source['checkpoint_sha256']
    summary=dict(family=family,source=source,policies={})
    try:
        for policy in POLICIES:
            item=report['families'][family]['configs'][policy];assert sha(item['path'])==item['sha256']
            out=root/(family+'_'+policy)
            if out.exists():raise FileExistsError('No automatic overwrite or retry')
            write_json(root/(family+'_status.json'),dict(state='running',policy=policy,updated_unix=time.time()))
            for cmd,cpu in commands(root,family,policy,source,item['path']):
                env=dict(os.environ)
                if cpu:env['CUDA_VISIBLE_DEVICES']=''
                with (root/(family+'_'+policy+('_replay' if cpu else '')+'.log')).open('xb') as log:
                    subprocess.run(cmd,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
            entry=evaluation(out)
            assert entry['manifest']['checkpoint_sha256']==source['checkpoint_sha256']
            assert entry['manifest']['sender_ranking']==policy
            if policy=='l1':
                diff={k:entry['evaluation']['ap'][k]-source['reference_ap'][k] for k in ('0.3','0.5','0.7')}
                assert max(abs(x) for x in diff.values())<=.0002,'L1 control AP gate failed'
                assert abs(entry['evaluation']['feature_packets']['mean_bytes']-source['reference_bytes']['mean_bytes'])<=.1
                entry['reference_ap_differences']=diff
            summary['policies'][policy]=entry
            write_json(root/(family+'_partial.json'),summary)
        summary['passed']=True
        write_json(root/(family+'_summary.json'),summary)
        write_json(root/(family+'_status.json'),dict(state='complete',updated_unix=time.time()))
    except Exception:
        write_json(root/(family+'_status.json'),dict(state='failed',error=traceback.format_exc(),note='No retry or termination of other families.'))
        raise


def launch(root,base,gpus):
    report=read(root/'audit.json');assert report['passed'] and sources(base)==report['sources']
    if len(gpus)!=3 or len(set(gpus))!=3:raise ValueError('Three distinct GPUs required')
    if (root/'launch.json').exists():raise FileExistsError('Already attempted')
    if shutil.disk_usage(str(root)).free<8*1024**3:raise RuntimeError('Insufficient reserve')
    for g in gpus:free_gpu(g)
    for family in FAMILIES:
        for item in report['families'][family]['configs'].values():assert sha(item['path'])==item['sha256']
    setup=dict(gpus=gpus,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),audit_sha256=sha(root/'audit.json'),started_unix=time.time())
    with (root/'launch.json').open('x') as f:json.dump(setup,f,indent=2)
    processes={}
    for family,gpu in zip(FAMILIES,gpus):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0')
        with (root/(family+'_supervisor.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_sender_rank','--root',str(root),'--family',family],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[family]=dict(pid=p.pid,gpu=gpu)
    write_json(root/'processes.json',processes);print(json.dumps(processes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    p.add_argument('--audit',action='store_true');p.add_argument('--family',choices=FAMILIES)
    p.add_argument('--gpus',nargs=3,type=int,default=[1,2,3]);a=p.parse_args();root=Path(a.root)
    if a.audit:audit(root,a.base)
    elif a.family:run_family(root,a.family)
    else:launch(root,a.base,a.gpus)
