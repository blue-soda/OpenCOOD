"""Full paired detector audit of a frozen, mathematically equivalent layout.

No new spatial history or masks are introduced: every spatial cell updates.
The separate full/cached views each maintain their own FM and deep anchors.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pointwise_spatial_state import ContextualPointwiseStage
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.tools.train_int_projected_gru import make_modes,evaluate
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


class FrozenLinearStage(torch.nn.Module):
    def __init__(self,stage):
        super().__init__();self.stage=stage

    def forward(self,temporal):
        current,evidence=self.stage.observe(temporal)
        old=torch.zeros_like(current);valid=torch.zeros_like(current[:,:1])
        mask=torch.ones_like(valid,dtype=torch.bool)
        return self.stage.linear_selected_update(current,evidence,old,valid,mask)


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    base,cfg,val,source=setup(a)
    assert digest(a.checkpoint)==a.sha256
    saved=torch.load(a.checkpoint,map_location='cpu');contract=saved['contract']
    assert contract['data']==source['contract'] and contract['source']['sha256']==source['checkpoint']['sha256']
    dependencies={}
    for name in ('memory','readout','repair'):
        entry=contract[name];assert digest(entry['checkpoint'])==entry['sha256']
        dependencies[name]=torch.load(entry['checkpoint'],map_location='cpu')
    assert contract['memory']['sha256']=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    assert contract['readout']['sha256']=='8bbf94e975997b2803c4a4c443fee2258589276e427dbc0a2550de6b562d7a0d'
    assert contract['repair']['sha256']=='9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'
    base.feature_memory=projected_gru(base.feature_memory,32)[0].cuda()
    base.feature_memory.load_state_dict(dependencies['memory']['memory'],strict=True)
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda()
    base.shrink_conv.load_state_dict(dependencies['readout']['readout'],strict=True)
    stage=ContextualPointwiseStage(base.backbone.resnet.layer0).cuda()
    stage.load_state_dict(saved['stage'],strict=True);base.backbone.resnet.layer0=stage
    student=copy.deepcopy(base)
    student.backbone.resnet.layer0=FrozenLinearStage(student.backbone.resnet.layer0)
    linear=student.backbone.resnet.layer0.stage
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda()
    repair=LightRepair(96,width=32).cuda();repair.load_state_dict(dependencies['repair']['repair'],strict=True)
    for module in (base,student,split,repair):
        module.eval()
        for p in module.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train']
    reference=json.loads((Path(a.checkpoint).parent/'report.json').read_text())
    expected=(reference['results']['full']['ap70'],reference['results']['cached']['ap70'])
    scope=('Frozen pointwise spatial checkpoint; base_full/base_cached use original Conv2d layout, '
        'full/cached use equivalent row-major linear layout. All spatial cells update, zero new spatial history. '
        'Existing deep cache period3 only; each view owns FM/deep states. Same-GPU rotating full-detector timing '
        'excludes IO/H2D/postprocess. Conv MACsx2 includes explicit functional linear MACs. '
        'This is a layout audit, not a regional reuse or convergence result.')
    write_json(out/'provenance.json',dict(source=source,checkpoint=a.checkpoint,sha256=a.sha256,contract=contract,
        epoch=saved['epoch'],expected_ap70=expected,scope=scope))
    modes=make_modes(base,student,split,repair,cfg);states={k:None for k in modes};costs={};max_error=0.
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));outputs={}
            for name,mode in modes.items():
                root=base if name.startswith('base') else student
                counter=ConvCounter(root,torch.nn.ModuleList([split,repair]));previous=states[name]
                try:pred,states[name],info=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                macs=counter.total+(0 if name.startswith('base') else linear.last_selected_linear_macs)
                key=name+('_partial' if info['local_partial'] else '_full')
                if key in costs:assert costs[key]==macs
                costs[key]=macs;outputs[name]=pred
                if info['local_partial']:assert states[name]['cache']['value'] is previous['cache']['value']
            for name in ('full','cached'):
                for key,value in outputs[name].items():
                    error=float((value-outputs['base_'+name][key]).abs().max());max_error=max(max_error,error)
                    assert torch.allclose(value,outputs['base_'+name][key],atol=1e-4,rtol=1e-4),(name,key,error)
    for name in ('full_full','cached_full','cached_partial'):assert costs[name]==costs['base_'+name]
    write_json(out/'engineering.json',dict(status='PASSED',frames=32,max_prediction_error=max_error,conv_macs=costs,
        owns_recurrent_states=True,immutable_own_deep_anchors=True,all_spatial_cells_update=True))
    directory=out/'validation';directory.mkdir()
    report=evaluate(base,student,split,repair,cfg,val,directory,costs,saved['epoch'],expected,scope)
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','checkpoint','sha256','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--worker',action='store_true')
    a=parser.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Pointwise layout audit exceeded2 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.evaluate_int_pointwise_layout']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
