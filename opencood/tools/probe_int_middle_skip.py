"""Screen one-step middle hold errors and measure the convolution support of real masks."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import TILE, compact_update, conv_costs, select_skip, supports, tile_sum
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.calibrate_int_middle_budget import detector,training_rows,aligned_previous,event_counts
from opencood.tools.check_int_deep_spike_stream import build_source,read
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.eval_int_middle_cap import TRAIN_COUNTS
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.probe_int_middle_motion import occupancy
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import trace_masks,local_request


def inspect(source,spatial,state,prev,meta,ego,engineering):
    fine,current=state['value'],state['coarse']
    h,w=current.shape[-2:]
    bounds=block_lidar_range(source.bounds,spatial.shape[-2:],(h,w),4)
    old,valid=warp_bev(prev['coarse'],prev['pose'],meta['pose'],bounds)
    coverage=warp_bev(torch.ones_like(old[:,:1]),prev['pose'],meta['pose'],bounds)[0].bool()
    eligible=tile_sum((~coverage).float())==0
    fbounds=block_lidar_range(source.bounds,spatial.shape[-2:],fine.shape[-2:],2)
    oldfine=warp_bev(prev['value'][:,::16],prev['pose'],meta['pose'],fbounds)[0]
    signal=(fine[:,::16]-oldfine).abs().mean(1,keepdim=True)
    fine_change=F.avg_pool2d(signal,3,2,1,count_include_pad=False)
    age=state['since_refresh']
    gain=.875 if age in (1,2,3) else 1.
    held={'zero':old,'phase':gain*old}
    err={k:(v-current).abs().sum(1,keepdim=True) for k,v in held.items()}
    scores={'fine':tile_sum(fine_change),'random':torch.zeros_like(tile_sum(fine_change)),
            'oracle_zero':tile_sum(err['zero']),'oracle_phase':tile_sum(err['phase'])}
    occupied=occupancy(ego,current).bool()
    rec=dict(frame=meta['frame'],scene=meta['scene'],age=age,sites=h*w,eligible=int(eligible.sum()),
             total_err={k:float(v.sum(dtype=torch.float64)) for k,v in err.items()},
             occupied=int(occupied.sum()),new_sites=int((~coverage).sum()),masks={})
    for how,cost in scores.items():
        skip=select_skip(cost,eligible,meta['frame'],(h,w))
        active=~skip
        assert not (skip & ~coverage).any()
        _,jobs,work,_,_=plan_host_coalesced_refresh(active,fine.shape[-2:],5,TILE)
        counts={k:int(v.sum()) for k,v in zip(('output','radius1','radius2'),supports(active))}
        x=dict(skipped=int(skip.sum()),layout_sites=work,layout_fallback=jobs is None,
               support=counts,occupied_skipped=int((skip&occupied).sum()),
               error={k:float((v*skip).sum(dtype=torch.float64)) for k,v in err.items()},
               occupied_error={k:float((v*skip*occupied).sum(dtype=torch.float64)) for k,v in err.items()})
        if engineering and how=='fine':
            dt=(meta['timestamp_us']-prev['timestamp_us'])/1e6
            result,actual=compact_update(source.core,fine,old,valid,dt,active,held['phase'])
            assert actual==counts
            ref=torch.where(active,current,held['phase'])
            x['max_abs_error']=float((result-ref).abs().max())
            x['actual_sites']=actual
            assert torch.allclose(result,ref,atol=1e-4,rtol=1e-5),(meta['frame'],x['max_abs_error'])
            assert torch.equal(result.masked_select(skip),ref.masked_select(skip))
        rec['masks'][how]=x
    return rec


@torch.no_grad()
def run(inputs,training,counts,out):
    torch.manual_seed(303)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    ref=read(inputs/'r297.json')
    for name,key in [('plan.json','plan'),('traces.json','traces'),('quantum.json','calibration')]:
        assert digest(inputs/name)==ref['inputs_sha256'][key]
    assert digest(counts)==TRAIN_COUNTS
    checks=[json.loads(l) for l in counts.read_text().splitlines()]
    plan=read(inputs/'plan.json')
    rows,traces,starts=training_rows(inputs,training,plan)
    cfg=load_yaml(str(inputs/'config.yaml'));cfg['data_augment']=[]
    source=build_source(inputs,cfg)
    assert model_digest(source)==ref['source_frozen_digest']
    q,_,_=load_quantum(inputs/'quantum.json',plan,512)
    model=detector(source,q,'carry',.25)
    cost=conv_costs(source.core)
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        train=digest(training/'train_plan.json'),trace=digest(inputs/'traces.json'),source=ref['source_frozen_digest'],
        reference=TRAIN_COUNTS,seed=303,tile=TILE,labels_used=0,params=sum(x.numel() for x in source.parameters()),
        conv_flops_per_site=cost,device=torch.cuda.get_device_name(),torch=torch.__version__))
    _,stream=loader(rows,cfg,0,303);n=0;clip=0
    with (out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,check in zip(stream,traces,checks):
            meta=sample['meta']
            assert trace['frame']==meta['frame']==check['frame'] and not meta['supervised']
            if meta['frame'] in starts:
                state=None;clip+=1;i=0
            prev=state
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            _,state,info=model.step_spatial(spatial,state,meta,local_request(trace_masks(trace,spatial.device)))
            assert info['full_refresh']==trace['refresh']==check['refresh']
            row=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'])
            if trace['refresh'] is None:
                assert event_counts(state['coarse'],aligned_previous(prev,state['coarse'],spatial,meta,source.bounds),.25)==check['carry'][0]
                row.update(inspect(source,spatial,state,prev,meta,ego,clip<=2 and i<12))
            log.write(json.dumps(row)+'\n');n+=1;i+=1
    assert n==len(rows)==256 and clip==8
    write_json(out/'report.json',dict(status='COMPLETED',scans=n,baseline_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','training','counts','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run(a.inputs,a.training,a.counts,a.output)
