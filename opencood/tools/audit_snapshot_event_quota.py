"""Real-input audit: no codec in optimization/validation; one encode on demand."""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_e3dsnn_single import checked_forward, write_json
from opencood.tools.audit_snapshot_native import seed
from opencood.utils.cp_optimizer import make_optimizer, apply_warmup
from opencood.utils import task_snapshot_packet as wire


def no_codec():
    stack=ExitStack()
    for name in ('select','encode','decode'):
        stack.enter_context(patch.object(wire,name,side_effect=AssertionError('Unexpected codec call: '+name)))
    stack.enter_context(patch('zlib.compress',side_effect=AssertionError('Unexpected compression')))
    return stack


def run(a):
    seed();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3)
    cfg=load_yaml(a.config);val=build_dataset(cfg,train=False);train=build_dataset(cfg,train=True)
    model=train_utils.create_model(cfg).cuda()
    initial=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(initial,strict=True)
    assert model.snapshot_selection=='event_quota'
    model.eval();model.packet_roundtrip=False;model.snapshot_packet_directory=None
    report=dict(frames=[],steps=[])
    for index in (0,1):
        sample=train_utils.to_device(val.collate_batch_test([val[index]]),torch.device('cuda'))
        for budget in (0,2304,8192):
            model.snapshot_budget=budget
            with no_codec(),torch.no_grad():
                torch.cuda.synchronize();started=time.perf_counter();direct=model(sample['ego'])
                torch.cuda.synchronize();seconds=time.perf_counter()-started
            assert 'packet_bytes' not in direct and not direct['communication_measured']
            for roundtrip in (False,True):
                model.snapshot_measure_bytes=True;model.packet_roundtrip=roundtrip
                with patch.object(wire,'encode',wraps=wire.encode) as enc, patch.object(wire,'decode',wraps=wire.decode) as dec, patch.object(wire,'select',side_effect=AssertionError('Prefix encoding')):
                    with torch.no_grad():measured=model(sample['ego'])
                assert enc.call_count==1 and dec.call_count==int(roundtrip)
                assert all(torch.equal(direct[k],measured[k]) for k in ('psm','rm','dm'))
                assert measured['communication_measured'] and len(measured['packet_bytes'])==1
                if budget==0:assert measured['packet_bytes']==[0]
            model.snapshot_measure_bytes=False;model.packet_roundtrip=False
            if budget==0:
                ego=dict(sample['ego'],disable_road=True)
                with no_codec(),torch.no_grad():local=model(ego)
                assert all(torch.equal(local[k],direct[k]) for k in ('psm','rm','dm'))
            report['frames'].append(dict(index=index,nominal_budget=budget,
                event_quota=model.snapshot_event_quotas.get(budget,budget),events=direct['stream_events'],
                measured_bytes=measured['packet_bytes'],direct_seconds=seconds,
                direct_encode_calls=0,measurement_encode_calls=1,measurement_decode_calls=0,
                roundtrip_encode_calls=1,roundtrip_decode_calls=1,outputs_bitwise_equal=True))
    model.load_state_dict(initial,strict=True);model.train();seed()
    optimizer=make_optimizer(cfg,model)
    for group in optimizer.param_groups:group['initial_lr']=group['lr']
    criterion=train_utils.create_loss(cfg).cuda()
    batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);apply_warmup(optimizer,0,step,2339,1)
        with no_codec():
            torch.cuda.synchronize();started=time.perf_counter()
            output,loss=checked_forward(model,criterion,batch);loss.backward()
            gradients={k:p.grad for k,p in model.named_parameters() if p.grad is not None}
            assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
            assert any(torch.count_nonzero(v)>0 for k,v in gradients.items() if k.startswith('content.project.'))
            if cfg['model']['core_method']=='e3dsnn_snapshot_pyramid':
                assert any(torch.count_nonzero(v)>0 for k,v in gradients.items() if k.startswith('joint_receiver.snapshot_pyramid.'))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert 'packet_bytes' not in output and not output['communication_measured']
        report['steps'].append(dict(step=step+1,loss=float(loss),gradient_norm=float(norm),
            gradient_tensors=len(gradients),seconds=seconds,encode_calls=0,decode_calls=0))
    assert any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
    report.update(passed=True,audit_updates_discarded=True,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest())
    write_json(Path(a.output),report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
