"""One bounded P+D study at actual snapshot bytes; sequential shared GPU jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from opencood.tools.evaluate_e3dsnn_ann_snapshot import SOURCES, FrozenSnapshot
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, sha, read, evaluation
from opencood.tools.launch_e3dsnn_joint_round import check_gpu
from opencood.tools.train_e3dsnn_single import validate, checked_forward, write_json
from opencood.models.fuse_modules.e3dsnn_snapshot_adapter import TRAINABLE

SEED=20260929
BUDGETS=(0,2048,4096,None)


def prepare(root):
    if root.exists():raise FileExistsError('New experiment root required')
    raw={f:yaml.safe_load(c.read_text()) for f,(_,c,_) in SOURCES.items()}
    for key in ('loss','optimizer','lr_scheduler','root_dir','validate_dir','data_manifest','data_augment'):
        assert raw['if8'].get(key)==raw['ann2bit'].get(key),key
    root.mkdir(parents=True)
    hashes={}
    for family,(checkpoint,_,digest) in SOURCES.items():
        assert sha(checkpoint)==digest
        folder=root/family;folder.mkdir()
        cfg=raw[family]
        cfg['model']['core_method']='e3dsnn_snapshot_if' if family=='if8' else 'e3dsnn_snapshot_ann'
        cfg['model']['args'].update(snapshot=dict(family=family,budget_bytes=2048),
            message_dropout=.1,packet_roundtrip=False,enable_ego_aux=False,ego_aux_weight=0.)
        cfg['train_params']=dict(batch_size=2,epoches=10)
        assert cfg['optimizer']['lr']==5e-5 and cfg['lr_scheduler']['step_size']==[7,9]
        for label,budget in [('train',2048)]+[('full' if b is None else 'b'+str(b),b) for b in BUDGETS]:
            current=copy.deepcopy(cfg);current['model']['args']['snapshot']['budget_bytes']=budget
            if label!='train':current['model']['args']['snapshot']['packet_directory']=str(folder/label/'packets')
            p=folder/(label+'.yaml');p.write_text(yaml.safe_dump(current));hashes[str(p.relative_to(root))]=sha(p)
    write_json(root/'protocol.json',dict(sources={f:dict(checkpoint=str(c),sha256=h) for f,(c,_,h) in SOURCES.items()},
        config_hashes=hashes,trainable=list(TRAINABLE),epochs=10,seed=SEED,training_budget_bytes=2048,
        selection='max validation BEV AP70 at 2048 bytes, epochs 0..10',final_budgets=list(BUDGETS),
        limitation='same additional training only; historical systems and selection/receivers differ',
        gradient='existing quantizer or fixed-frontier STE; sorting and actual bytes not differentiable',
        objective='one fused detection loss; frozen ego auxiliary disabled identically; no distillation/rate loss',
        exit='one two-family P+D round; no extra LR/epochs/codec sweep'))


def audit(root,family):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    torch.cuda.set_per_process_memory_fraction(.2)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    cfg=load_yaml(str(root/family/'train.yaml'))
    checkpoint,_,digest=SOURCES[family];assert sha(checkpoint)==digest
    model=train_utils.create_model(cfg).cuda().eval()
    initial=torch.load(str(checkpoint),map_location='cpu');model.load_state_dict(initial,strict=True)
    trainable=[k for k,p in model.named_parameters() if p.requires_grad]
    assert set(trainable)==set(TRAINABLE)
    out=root/family/'audit';out.mkdir(exist_ok=False)
    report=dict(family=family,source_sha256=digest,trainable=trainable,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),original_comparisons=[])
    val=build_dataset(cfg,train=False)
    # Original reference and new forward share every encoder output.
    for i in (0,1):
        batch=train_utils.to_device(val.collate_batch_test([val[i]]),torch.device('cuda'));data=batch['ego']
        original_local,original_road=model.encode_membrane,model.encode_lidar
        with torch.no_grad():
            local,membrane=original_local(data['processed_lidar'],1);road=original_road(data['road_lidar'],1)
            model.encode_membrane=lambda lidar,n:(local,membrane);model.encode_lidar=lambda lidar,n:road
            for mode,budget in (('b2048',2048),('full',None)):
                folder=out/(str(i)+'_'+mode);folder.mkdir();(folder/'packets').mkdir()
                reference=FrozenSnapshot(model,family,mode,folder,False).eval()
                expected=reference(data)
                model.snapshot_budget=budget;model.packet_roundtrip=True;model.snapshot_audit=True
                actual=model(data)
                assert all(torch.equal(actual[k],expected[k]) for k in ('psm','rm','dm'))
                assert actual['packet_bytes']==expected['packet_bytes']
                assert model.snapshot_records[0]['packet_sha256']==reference.records[0]['packet_sha256']
                report['original_comparisons'].append(dict(frame=i,mode=mode,heads_and_packet_exact=True))
        model.encode_membrane,model.encode_lidar=original_local,original_road
    training=build_dataset(cfg,train=True)
    batch=train_utils.to_device(training.collate_batch_train([training[0],training[1]]),torch.device('cuda'))
    data=batch['ego'];local_fn,road_fn=model.encode_membrane,model.encode_lidar
    with torch.no_grad():
        local,membrane=local_fn(data['processed_lidar'],2);road=road_fn(data['road_lidar'],2)
        model.encode_membrane=lambda lidar,n:(local,membrane);model.encode_lidar=lambda lidar,n:road
        report['augmented_wire']=[]
        for budget in (0,2048,4096):
            model.snapshot_budget=budget;model.packet_roundtrip=False;direct=model(data)
            model.packet_roundtrip=True;decoded=model(data)
            assert all(torch.equal(direct[k],decoded[k]) for k in ('psm','rm','dm'))
            assert all(n<=budget for n in decoded['packet_bytes'])
            report['augmented_wire'].append(dict(budget=budget,bytes=decoded['packet_bytes'],exact=True))
            if budget==0:
                model.fusion_mode='ego';ego=model(data);model.fusion_mode='residual'
                assert all(torch.equal(ego[k],decoded[k]) for k in ('psm','rm','dm'))
                report['zero_equals_ego']=True
    model.encode_membrane,model.encode_lidar=local_fn,road_fn
    model.snapshot_budget=2048;model.snapshot_audit=False;model.packet_roundtrip=False
    model.train();model.message_dropout=0.
    criterion=train_utils.create_loss(cfg).cuda()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    tic=time.perf_counter();opt.zero_grad(set_to_none=True)
    _,loss=checked_forward(model,criterion,batch);loss.backward()
    gradients={k:float(p.grad.abs().sum()) for k,p in model.named_parameters() if p.grad is not None}
    assert set(gradients)==set(TRAINABLE) and all(np.isfinite(g) and g>0 for g in gradients.values())
    opt.step();torch.cuda.synchronize();step_time=time.perf_counter()-tic
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])]
    assert set(changed)==set(TRAINABLE),changed
    assert step_time<=20.,step_time
    # Discard the step before the bounded real-data timing gate.
    model.load_state_dict(initial,strict=True);model.packet_roundtrip=True;model.message_dropout=.1
    result=validate(model,val,2,limit=64,profile=True)
    write_json(out/'audit64_metrics.json',result)
    assert result['samples']==64 and result['feature_packets']['max_bytes']<=2048
    assert result['profile']['milliseconds']['forward_ms']['mean']<=5000.
    report.update(passed=True,gradient_l1=gradients,optimizer_step_discarded=True,
        other_state_unchanged=True,training_step_seconds=step_time,audit64=result,
        source_checkpoint_unchanged=sha(checkpoint)==digest)
    write_json(out/'audit.json',report)
    print(json.dumps(dict(family=family,passed=True,gradients=gradients,step_seconds=step_time)),flush=True)


def check_frozen(root,family):
    initial=torch.load(str(SOURCES[family][0]),map_location='cpu');reports={}
    for name in ('last.pt','best_ap70.pth'):
        p=root/family/'train'/name;state=torch.load(str(p),map_location='cpu')
        if name=='last.pt':assert state['epoch']==10;state=state['model']
        changed=[k for k,v in state.items() if not torch.equal(v,initial[k])]
        assert set(changed)<=set(TRAINABLE)
        if name=='last.pt':assert set(changed)==set(TRAINABLE)
        reports[name]=dict(sha256=sha(p),changed=changed)
    write_json(root/family/'frozen.json',dict(passed=True,checkpoints=reports))


def coordinate(root,gpu):
    try:
        protocol=read(root/'protocol.json')
        for path,digest in protocol['config_hashes'].items():assert sha(root/path)==digest
        def run(name,command,timeout,gpu_job=True):
            if gpu_job:
                admitted=False
                for attempt in range(3):
                    try:
                        admission=check_gpu(gpu,True,.2);admitted=True;break
                    except RuntimeError as error:
                        if 'shared admission failed' not in str(error):raise
                        write_json(root/'status.json',dict(state='waiting_resource',stage=name,error=str(error)))
                        if attempt<2:time.sleep(60)
                if not admitted:raise RuntimeError('Resource admission exhausted; no job started')
                write_json(root/(name+'_admission.json'),admission)
            write_json(root/'status.json',dict(state='running',stage=name,started_unix=time.time()))
            env=dict(os.environ)
            if not gpu_job:env['CUDA_VISIBLE_DEVICES']=''
            with (root/(name+'.log')).open('xb') as log:
                subprocess.run(command,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,check=True,timeout=timeout)
        for family in SOURCES:
            run('audit_'+family,[sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_snapshot_adapt',
                '--root',str(root),'--audit',family],1800)
            assert read(root/family/'audit/audit.json')['passed']
        write_json(root/'audits_complete.json',dict(passed=True))
        for family in SOURCES:
            folder=root/family;source=SOURCES[family][0]
            base=[sys.executable,'-u','-m','opencood.tools.train_e3dsnn_single','--workers','2','--seed',str(SEED)]
            run('train_'+family,base+['--mode','train','--config',str(folder/'train.yaml'),
                '--checkpoint',str(source),'--output',str(folder/'train'),'--epochs','10',
                '--selection-iou','0.7','--retain-initial-best'],43200)
            check_frozen(root,family)
            result=dict(passed=True,family=family,initial=read(folder/'train/initial_metrics.json'),
                best=read(folder/'train/best_ap70_metrics.json'),frozen=read(folder/'frozen.json'),evaluations={})
            for budget in BUDGETS:
                name='full' if budget is None else 'b'+str(budget);output=folder/name
                run(family+'_'+name,base+['--mode','eval','--config',str(folder/(name+'.yaml')),
                    '--checkpoint',str(folder/'train/best_ap70.pth'),'--output',str(output),
                    '--eval-3d','--save-predictions','--profile','--packet-roundtrip'],14400)
                run(family+'_'+name+'_replay',[sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions',
                    '--evaluation-dir',str(output)],3600,False)
                row=evaluation(output)
                records=[json.loads(s) for s in (output/'packets/index.jsonl').read_text().splitlines()]
                index=read(output/'predictions/index.json')['frames'];assert len(records)==len(index)==1738
                for packet,frame in zip(records,index):
                    p=output/'packets'/packet['file']
                    assert sha(p)==packet['sha256'] and p.stat().st_size==packet['actual_bytes']==frame['communication']['total_bytes']
                    assert packet['frame_id']==frame['frame_id']
                    if budget is not None:assert packet['actual_bytes']<=budget
                assert row['manifest']['checkpoint_sha256']==sha(folder/'train/best_ap70.pth')
                result['evaluations'][name]=row
            write_json(folder/'summary.json',result)
        write_json(root/'summary.json',dict(passed=True,protocol=protocol,
            results={f:read(root/f/'summary.json') for f in SOURCES}))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'))
        raise


def launch(root,gpu):
    admission=check_gpu(gpu,True,.2)
    prepare(root)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCOOD_CUDA_MEMORY_FRACTION='0.2')
    with (root/'pipeline.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.launch_e3dsnn_snapshot_adapt',
            '--root',str(root),'--coordinate','--gpu',str(gpu)],cwd=str(REPO),env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,admission=admission,started_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip())
    write_json(root/'launch.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--coordinate',action='store_true');p.add_argument('--audit',choices=tuple(SOURCES))
    a=p.parse_args()
    if a.audit:audit(Path(a.root),a.audit)
    elif a.coordinate:coordinate(Path(a.root),a.gpu)
    else:launch(Path(a.root),a.gpu)
