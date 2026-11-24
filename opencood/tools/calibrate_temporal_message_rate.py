"""Match the number of backbone event messages on training clips, using no AP or labels."""
import argparse
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_planar_window import attach_window
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.check_int_silent_trajectory import scene_windows
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid
from opencood.tools.train_int_full_dair_pair import full_rows
from opencood.data_utils.int_v2v_ego_stream import v2v_rows


@torch.no_grad()
def calibrate(a):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(a.seed)
    cfg=fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment']=[]
    manifest=a.root/('sequence_manifest.json' if a.domain=='dair' else 'formal_splits/v2v_official_deduplicated.jsonl')
    train=full_rows(manifest,a.dataset,'train') if a.domain=='dair' else v2v_rows(manifest,'train',a.dataset)
    rows=attach_window(scene_windows(train),False)
    for row in rows:
        row['supervised']=False
    assert len(rows)%5==0
    base=a.root/(('full_temporal' if a.domain=='dair' else 'v2v_full_temporal') if a.arm=='native' else 'qat549_303_fit')/'final.pt'
    dropped=a.root/('retrain554_%s_fit'%a.arm)/'final.pt'
    base=a.carry_checkpoint or base
    dropped=a.discard_checkpoint or dropped
    source=fit.build_source(a.root/'inputs',cfg)
    net=RecoveryPyramid(source.core,a.root/'initial.pt','fine_temporal').cuda().eval()
    stream=fit.loader(rows,cfg,2,a.seed)[1]
    counts=torch.zeros(2,dtype=torch.int64,device='cuda')
    def observe(module,args,out):
        counts[int(torch.is_tensor(args[1]))]+=torch.count_nonzero(out[0])
    for sender in net.senders:
        sender.register_forward_hook(observe)
        if a.arm=='qat':
            sender.continuous_threshold=.5
            sender.payload_dtype='int8'
    # Cache only one five-frame input window on-device. Each pass starts at its true refresh.
    grid=[None,.505,.51,.52,.53,.54,.55,.57,.60,.65,.70,.80]
    fits=[torch.load(c,map_location='cpu') for c in (dropped,base)]
    assert fits[1]['contract']['manifest']==fit.digest(manifest)
    carry_contract=dict(fits[1]['contract'])
    assert carry_contract['seed']==a.seed
    if a.domain=='dair' and 'dataset' not in carry_contract:
        assert carry_contract['manifest']=='184798f833dd52b6eee2cf2f3b04ab7684221be43b9302edccd4889f51d88738'
        carry_contract['dataset']='dair_single'
    assert fits[0]['contract']==dict(carry_contract,train_drop_silent=True)
    assert (fits[1]['contract'].get('training_payload')=='int8')==(a.arm=='qat')
    totals=[[0,0] for _ in range(len(grid)+1)]
    frames=[]
    for i,sample in enumerate(stream):
        row=rows[i]
        assert (i%5==0)==row['reset_before'] and sample['meta']['frame']==row['frame']
        ego=fit.to_device(sample['ego'],torch.device('cuda'))
        frames.append((fit.spatial_input(source,ego),row['window_ordinary']))
        if len(frames)!=5:
            continue
        for f,parent in enumerate(fits):
            net.load_state_dict(parent['net'])
            for sender in net.senders:
                sender.drop_silent_remainder=(f==0)
            for j,floor in enumerate([None] if f==0 else grid):
                for sender in net.senders:
                    sender.ordinary_firing_floor=floor
                counts.zero_()
                state=None
                for spatial,ordinary in frames:
                    _,state,_=net(spatial,source.core,state,ordinary=ordinary)
                record=totals[0 if f==0 else 1+j]
                for k,v in enumerate(counts.tolist()):
                    record[k]+=v
        frames.clear()
    selected=min(range(len(grid)),key=lambda j:abs(sum(totals[j+1])-sum(totals[0])))
    result=dict(domain=a.domain,arm=a.arm,seed=a.seed,scans=len(rows),train_manifest=fit.digest(manifest),
                calibration_frames=[(r['scene'],r['frame']) for r in rows],
                carry_sha=fit.digest(base),discard_sha=fit.digest(dropped),
                grid=grid,discard_refresh_ordinary=totals[0],carry_refresh_ordinary=totals[1:],
                selected_ordinary_floor=grid[selected],train_relative_difference=sum(totals[selected+1])/sum(totals[0])-1)
    assert len({v[0] for v in totals[1:]})==1
    with a.out.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','dataset','out'):
        p.add_argument(name,type=Path)
    p.add_argument('domain',choices=('dair','v2v'))
    p.add_argument('arm',choices=('native','qat'))
    p.add_argument('--seed',type=int,default=303,choices=(303,707))
    p.add_argument('--carry-checkpoint',type=Path)
    p.add_argument('--discard-checkpoint',type=Path)
    calibrate(p.parse_args())
