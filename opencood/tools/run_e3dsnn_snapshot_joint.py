"""Bounded fixed-budget versus multi-budget Scheme-2 joint adaptation."""
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
from opencood.tools.launch_e3dsnn_hybrid_round import REPO, sha, read, evaluation
from opencood.tools.train_e3dsnn_single import write_json, checked_forward

SOURCE_SHA = 'c51a33f3c1d935a8528763ddbb68ccb3298475beab842a49c5a71216c56b5a78'
ARMS = {'fixed2k': [2048], 'multi248k': [2048, 4096, 8192]}
BUDGETS = (0, 2048, 4096, 8192)
SEED = 20260929


def prepare(root):
    assert sha(root/'source.pth') == SOURCE_SHA
    source = yaml.safe_load((root/'source.yaml').read_text())
    data = '/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure'
    for key in ('root_dir', 'validate_dir', 'test_dir'):
        source[key] = data + '/' + Path(source[key]).name
    source['data_dir'] = data
    source['data_manifest'] = '/data1/zhh/gzc/diagnostics/e3dsnn_event_utility_source_20261022/pair_manifest.json'
    hashes = {}
    for arm, budgets in ARMS.items():
        folder = root/arm
        folder.mkdir(exist_ok=False)
        cfg = copy.deepcopy(source)
        cfg['model']['core_method'] = 'e3dsnn_snapshot_joint'
        cfg['model']['args']['snapshot_joint'] = dict(train_budgets=budgets, ego_aux_weight=.5)
        cfg['model']['args']['message_dropout'] = .1
        cfg['train_params'].update(batch_size=2, epoches=10)
        cfg['optimizer']['lr'] = 5e-5
        cfg['lr_scheduler']['step_size'] = [7, 9]
        (folder/'train.yaml').write_text(yaml.safe_dump(cfg))
        hashes[str((folder/'train.yaml').relative_to(root))] = sha(folder/'train.yaml')
        for budget in BUDGETS:
            val = copy.deepcopy(cfg)
            val['model']['args']['snapshot']['budget_bytes'] = budget
            val['model']['args']['snapshot']['packet_directory'] = str(folder/('b'+str(budget))/'packets')
            path = folder/('b'+str(budget)+'.yaml')
            path.write_text(yaml.safe_dump(val));hashes[str(path.relative_to(root))] = sha(path)
    write_json(root/'protocol.json', dict(source_sha256=SOURCE_SHA, seed=SEED,
        config_hashes=hashes, arms=ARMS, epochs=10, batch_size=2,
        criterion='Ldet(sampled budget) + 0.5 Ldet(local); no KD, no rate proxy',
        trainable='3D conv4/conv_out, project, expand, local gate/current_scale, ANN BEV and continuous heads',
        frozen='early 3D, all BN buffers, unused legacy fusion/order parameters',
        selection='same 2048-byte validation AP70, epoch 1..10; historical source retained separately',
        exit='No extension: if no >=0.5 AP point gain at 2048 bytes, close this joint-scope hypothesis',
        final_budgets=BUDGETS, source_ap70=.5823401,
        dataset_hashes={k:sha(source[k]) for k in ('root_dir','validate_dir','data_manifest')}))


def audit(root, arm):
    from opencood.tools import train_utils
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.models.e3dsnn_snapshot_if import E3dsnnSnapshotIf
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(.6)
    cfg = load_yaml(str(root/arm/'train.yaml'))
    dataset = build_dataset(cfg, train=True)
    val = build_dataset(cfg, train=False)
    assert len(dataset)==4678 and len(val)==1738
    model = train_utils.create_model(cfg).cuda()
    state = torch.load(str(root/'source.pth'), map_location='cpu')
    model.load_state_dict(state, strict=True)
    reference = E3dsnnSnapshotIf(cfg['model']['args']).cuda()
    reference.load_state_dict(state, strict=True)
    batch = train_utils.to_device(val.collate_batch_test([val[0]]), torch.device('cuda'))
    model.eval();reference.eval(); rows=[]
    with torch.no_grad():
        for budget in BUDGETS:
            model.snapshot_budget = reference.snapshot_budget = budget
            model.packet_roundtrip = False
            direct = model(batch['ego']); old = reference(batch['ego'])
            assert all(torch.equal(direct[k], old[k]) for k in ('psm','rm','dm'))
            model.packet_roundtrip = True
            wire = model(batch['ego'])
            assert all(torch.equal(direct[k], wire[k]) for k in ('psm','rm','dm'))
            assert direct['packet_bytes']==wire['packet_bytes']==old['packet_bytes']
            assert max(wire['packet_bytes'])<=budget
            rows.append(dict(budget=budget,bytes=wire['packet_bytes'],exact=True))
        model.fusion_mode='ego';ego=model(batch['ego']);model.fusion_mode='residual'
        model.snapshot_budget=0;zero=model(batch['ego'])
        assert all(torch.equal(ego[k],zero[k]) for k in ('psm','rm','dm'))
    del reference,batch
    model.packet_roundtrip=False; model.snapshot_budget=2048
    model.train();fixed=model.fixed_state_digest()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    batch=train_utils.to_device(dataset.collate_batch_train([dataset[0],dataset[1]]),torch.device('cuda'))
    criterion=train_utils.create_loss(cfg).cuda()
    optimizer=train_utils.setup_optimizer(cfg,model)
    model.message_dropout=0
    gradients={};losses=[]
    # Audit each candidate budget with real task gradients; discard all updates.
    registered=model.joint_train_budgets
    for budget in registered:
        model.joint_train_budgets=(budget,)
        optimizer.zero_grad(set_to_none=True)
        output,loss=checked_forward(model,criterion,batch)
        loss.backward()
        groups=('backbone_3d.conv4.','backbone_3d.conv_out.','backbone_2d.',
                'cls_head.','reg_head.','dir_head.','content.project.','content.expand.','joint_receiver.')
        grad={p:sum(float(v.grad.abs().sum()) for n,v in model.named_parameters()
                    if n.startswith(p) and v.grad is not None) for p in groups}
        assert all(np.isfinite(v) and v>0 for v in grad.values()),grad
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step();assert model.fixed_state_digest()==fixed
        gradients[str(budget)]=grad;losses.append(float(loss.detach()))
    changed=[n for n,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[n])]
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    assert changed and set(changed)<=trainable
    write_json(root/arm/'audit.json',dict(passed=True,wire_parity=rows,zero_equals_ego=True,
        source_sha256=SOURCE_SHA,gradients=gradients,losses=losses,changed=changed,
        fixed_sha256=fixed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        note='All audit updates discarded; training reloads pristine source'))


def coordinate(root,arm):
    folder=root/arm
    try:
        protocol=read(root/'protocol.json')
        assert sha(root/'source.pth')==SOURCE_SHA
        for rel,digest in protocol['config_hashes'].items():assert sha(root/rel)==digest
        def run(name,command,timeout=172800,cpu=False):
            write_json(folder/'status.json',dict(state='running',stage=name,unix=time.time()))
            env=dict(os.environ)
            if cpu:env['CUDA_VISIBLE_DEVICES']=''
            with (folder/(name+'.log')).open('x') as log:
                subprocess.run(command,cwd=str(REPO),env=env,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,check=True,timeout=timeout)
        run('audit',[sys.executable,'-u','-m',__spec__.name,'--root',str(root),'--arm',arm,'--mode','audit'],1800)
        assert read(folder/'audit.json')['passed']
        base=[sys.executable,'-u','-m','opencood.tools.train_e3dsnn_single','--workers','2','--seed',str(SEED)]
        run('train',base+['--mode','train','--config',str(folder/'train.yaml'),
            '--checkpoint',str(root/'source.pth'),'--output',str(folder/'train'),
            '--epochs','10','--selection-iou','0.7'])
        result={}
        for budget in BUDGETS:
            name='b'+str(budget);out=folder/name
            run(name,base+['--mode','eval','--config',str(folder/(name+'.yaml')),
                '--checkpoint',str(folder/'train/best_ap70.pth'),'--output',str(out),
                '--packet-roundtrip','--eval-3d','--save-predictions','--profile'],21600)
            run(name+'_replay',[sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions',
                '--evaluation-dir',str(out)],3600,True)
            result[name]=evaluation(out)
            records=[json.loads(s) for s in (out/'packets/index.jsonl').read_text().splitlines()]
            assert len(records)==1738
            for p,f in zip(records,read(out/'predictions/index.json')['frames']):
                packet=out/'packets'/p['file']
                assert sha(packet)==p['sha256']
                assert packet.stat().st_size==p['actual_bytes']==f['communication']['total_bytes']
                assert p['actual_bytes']<=budget and p['frame_id']==f['frame_id']
        write_json(folder/'summary.json',result)
        write_json(folder/'status.json',dict(state='complete',sha256=sha(folder/'summary.json')))
    except Exception:
        write_json(folder/'status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False))
        raise


def launch(root,arm,gpu):
    folder=root/arm
    # Exclusive launch marker protects against duplicate heartbeat starts.
    marker=folder/'launch.json'
    with marker.open('x') as f:f.write(json.dumps(dict(state='admitting')))
    readings=[]
    for _ in range(5):
        row=subprocess.check_output(['nvidia-smi','-i',str(gpu),
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
        free,util=map(int,row.split(','));readings.append(dict(free_mib=free,util=util))
        if free<16000 or util>20:raise RuntimeError('GPU admission rejected: '+row)
        time.sleep(2)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCOOD_CUDA_MEMORY_FRACTION='.6')
    with (folder/'pipeline.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--root',str(root),
            '--arm',arm,'--mode','coordinate'],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,gpu=gpu,admission=readings,unix=time.time(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip())
    write_json(marker,record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--gpu',type=int)
    p.add_argument('--mode',choices=('prepare','launch','audit','coordinate'),required=True)
    a=p.parse_args();root=Path(a.root)
    if a.mode=='prepare':prepare(root)
    elif a.mode=='launch':launch(root,a.arm,a.gpu)
    elif a.mode=='audit':audit(root,a.arm)
    else:coordinate(root,a.arm)
