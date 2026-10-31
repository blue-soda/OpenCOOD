"""Real-data byte/output/gradient/update identity for the CPU codec replacement."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward, validate, write_json
from opencood.utils.cp_optimizer import make_optimizer, apply_warmup


def seed():
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)


def run(a):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load_yaml(a.config);val=build_dataset(cfg,train=False);train=build_dataset(cfg,train=True)
    model=train_utils.create_model(cfg).cuda();initial=torch.load(a.checkpoint,map_location='cpu')
    model.load_state_dict(initial,strict=True);model.eval();model.packet_roundtrip=True;report={'frames':[],'steps':[]}
    for index in (0,1):
        sample=train_utils.to_device(val.collate_batch_test([val[index]]),torch.device('cuda'))
        records=[]
        for native in (False,True):
            os.environ['OPENCOOD_SNAPSHOT_NATIVE_LIB']=a.library if native else ''
            packets=[];model._snapshot_packet_observer=lambda fid,p:packets.append(hashlib.sha256(p).hexdigest())
            start=time.perf_counter()
            with torch.no_grad():output=model(sample['ego'])
            records.append(({k:output[k].detach().clone() for k in ('psm','rm','dm')},packets,output['packet_bytes'],time.perf_counter()-start))
        assert records[0][1:3]==records[1][1:3]
        assert all(torch.equal(records[0][0][k],records[1][0][k]) for k in records[0][0])
        report['frames'].append(dict(index=index,packet_sha256=records[1][1],bytes=records[1][2],reference_seconds=records[0][3],native_seconds=records[1][3],exact=True))
        print(json.dumps(report['frames'][-1]),flush=True)
    model._snapshot_packet_observer=None
    model.packet_roundtrip=False
    before=(torch.get_rng_state(),np.random.get_state(),random.getstate(),torch.cuda.get_rng_state_all())
    validate(model,val,2,limit=1)
    after=torch.get_rng_state();torch.set_rng_state(before[0]);torch.empty((),dtype=torch.int64).random_()
    assert torch.equal(after,torch.get_rng_state())
    npafter=np.random.get_state();assert before[1][0]==npafter[0] and np.array_equal(before[1][1],npafter[1]) and before[1][2:]==npafter[2:]
    assert before[2]==random.getstate() and all(torch.equal(x,y) for x,y in zip(before[3],torch.cuda.get_rng_state_all()))
    report['parent_rng_matches_deferred_base_seed_draw']=True
    seed();batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
    criterion=train_utils.create_loss(cfg).cuda();reference=[]
    for native in (False,True):
        os.environ['OPENCOOD_SNAPSHOT_NATIVE_LIB']=a.library if native else ''
        model.load_state_dict(initial,strict=True);model.train();optimizer=make_optimizer(cfg,model);seed()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);apply_warmup(optimizer,0,step,2339,1)
            output,loss=checked_forward(model,criterion,batch);loss.backward()
            gradients={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if not native:reference.append((float(loss),gradients,state))
            else:
                expected=reference[step];assert float(loss)==expected[0]
                assert set(gradients)==set(expected[1])
                assert all(torch.equal(v,expected[1][k]) for k,v in gradients.items())
                assert all(torch.equal(v,expected[2][k]) for k,v in state.items())
                report['steps'].append(dict(step=step+1,loss=float(loss),gradient_tensors=len(gradients),state_tensors=len(state),gradient_norm=float(norm),exact=True))
            print(json.dumps(dict(native=native,step=step+1,loss=float(loss))),flush=True)
    report.update(passed=True,audit_updates_discarded=True,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest())
    write_json(Path(a.output),report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','library','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
