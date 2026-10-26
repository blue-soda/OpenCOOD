"""Bounded train-prefix diagnostic, with no AP, latency or savings claim."""
import math
from pathlib import Path
import torch
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.models.sub_modules.int_current_readout import current_readout_from_mature
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.models.sub_modules.int_short_readout import ShortReadoutState
from opencood.models.sub_modules.int_readout_diagnostic import DiagnosticReadoutState, readout_error
from opencood.models.sub_modules.int_anchor_residual import predict
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_gpu_exclusivity import sample_gpu_usage


def diagnose(a, cfg, train, baseline, out, deep_repair, checkpoint, source):
    path=Path(a.short_readout_diagnostic)
    saved=torch.load(str(path),map_location='cpu');contract=saved['contract']
    assert contract['source']=={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    assert contract['frozen_deep_checkpoint']==checkpoint
    assert contract['deep_period']==4 and contract['readout_period']==3
    if contract.get('kind','residual')=='residual':
        repair=LightRepair(256,contract['width'],current_channels=96).cuda().eval()
    else:
        repair,_=current_readout_from_mature(baseline.readout,contract['width'],contract['kind']=='current_gated')
        repair=repair.cuda().eval()
    repair.load_state_dict(saved['repair'],strict=True)
    for p in repair.parameters():p.requires_grad_(False)
    actual=RecurrentDeepDetector(baseline,deep_repair,4)
    replay=RecurrentDeepDetector(baseline,deep_repair,4)
    actual.readout_cache=DiagnosticReadoutState(actual.readout,repair,actual.bounds,3)
    replay.readout_cache=ShortReadoutState(replay.readout,repair,replay.bounds,3)
    write_json(out/'diagnostic_contract.json',dict(checkpoint=str(path),sha256=digest(path),epoch=saved['epoch'],
        width=contract['width'],frames=512,split='train prefix; no fitting',
        comparison='Same real recursive upstream/current input: recursive readout vs repair with previous full readout as history vs warped previous full readout alone.',
        causality='Teacher full output never enters returned student state/confidence. Teacher history is diagnostic only; cannot deploy it without full readout each frame.',
        controls='Real recursive prediction and complete readout state replayed against uninstrumented inference on EVERY frame.',
        metrics='Global and teacher-confidence>=.1 foreground relative RMSE, split by depth1/2. Pooled squared error/target energy. No AP or runtime/savings claims.'))
    totals={};counts={};state=None;control=None;refresh=0;maximum=0.;sharing=[]
    _,stream=loader(train[:512],cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            pred,state,info=actual.step(ego,state,meta)
            other,control,_=replay.step(ego,control,meta)
            for k in pred:
                assert torch.equal(pred[k],other[k]),'Diagnostic changed actual prediction'
                maximum=max(maximum,float((pred[k]-other[k]).abs().max()))
            assert torch.equal(state['readout']['value'],control['readout']['value'])
            observation=actual.readout_cache.observation
            if observation is None:refresh+=1
            else:
                depth=str(observation['depth']);counts[depth]=counts.get(depth,0)+1
                teacher=predict(actual.model,observation['target'])
                foreground=teacher['psm'].sigmoid().amax(1,keepdim=True)>=.1
                for name in ('recursive','teacher_history','teacher_warp_only'):
                    for region,mask in (('all',None),('foreground',foreground)):
                        key=depth+'/'+name+'/'+region
                        sample_error=readout_error(observation[name],observation['target'],mask)
                        total=totals.setdefault(key,{k:0. for k in sample_error})
                        for k,v in sample_error.items():total[k]+=v
            if (index+1)%100==0:
                sharing.append(sample_gpu_usage(a.gpu))
                write_json(out/'status.json',dict(status='RUNNING',phase='diagnostic',scans=index+1))
    assert index+1==512 and set(counts)=={'1','2'}
    metrics={}
    for key,total in totals.items():
        metrics[key]=dict(total,rmse=math.sqrt(total['squared_error']/total['elements']) if total['elements'] else None,
            relative_rmse=math.sqrt(total['squared_error']/total['target_energy']) if total['target_energy'] else None)
    report=dict(status='COMPLETED',scans=512,refresh_frames=refresh,ordinary_by_depth=counts,
        prediction_max_error=maximum,actual_trajectory_exact=True,metrics=metrics,gpu_samples=sharing,
        note='Offline conditional feature diagnostic; no teacher-history AP, latency or FLOPs benefit claimed.')
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED',scans=512))
