"""Compare two senders on identical local targets without feeding diagnostic events back into the network."""
import argparse
import copy
import json
from pathlib import Path

import torch

from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.models.sub_modules.int_planar_window import attach_window
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic
from opencood.tools.train_int_full_dair_pair import full_rows


def scene_windows(rows):
    windows = {}
    clip = []
    for row in periodic(rows):
        if row['reset_before']:
            clip = []
        clip.append(row)
        if len(clip) == 5:
            windows.setdefault(row['scene'],list(clip))
    return [r for window in windows.values() for r in window]


@torch.no_grad()
def run(root, datasets, domain, arm, out):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    manifest = root/('sequence_manifest.json' if domain=='dair' else 'formal_splits/v2v_official_deduplicated.jsonl')
    rows = full_rows(manifest,datasets,'train') if domain=='dair' else v2v_rows(manifest,'train')
    all_scenes = set(r['scene'] for r in rows)
    rows = attach_window(scene_windows(rows),False)
    cfg = fit.load_yaml(str(root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(root/'inputs',cfg)
    net = RecoveryPyramid(source.core,root/'initial.pt','fine_temporal').cuda().eval()
    native = 'full_temporal' if domain=='dair' else 'v2v_full_temporal'
    ck = root/(native if arm=='native' else 'qat549_303_fit')/'final.pt'
    parent = torch.load(ck,map_location='cpu')
    assert parent['contract']['seed']==303 and parent['contract']['manifest']==fit.digest(manifest)
    net.load_state_dict(parent['net'])
    columns = ['elements','carry_abs_error_in_steps','discard_abs_error_in_steps','carry_events','discard_events']
    records = []
    buckets = torch.zeros((5,3,len(columns)),dtype=torch.float64,device='cuda')
    phase = 0
    def register(sender,stage):
        if arm=='qat':
            assert parent['contract']['training_payload']=='int8'
            sender.continuous_threshold=.5
            sender.payload_dtype='int8'
        shadow=copy.deepcopy(sender)
        shadow.drop_silent_remainder=True
        other_received=None
        def observe(module,args,output):
            nonlocal other_received
            target,prior=args
            shadow.step_multiplier=module.step_multiplier
            old=other_received if torch.is_tensor(prior) else 0.
            discarded,other_received,_=shadow(target,old)
            sent,received,_=output
            if phase==0:
                assert not torch.is_tensor(prior) and torch.equal(sent,discarded)
            q=module.step()
            dest=buckets[phase,stage]
            dest[0]+=target.numel()
            dest[1]+=((target-received).abs()/q).sum(dtype=torch.float64)
            dest[2]+=((target-other_received).abs()/q).sum(dtype=torch.float64)
            dest[3]+=torch.count_nonzero(sent)
            dest[4]+=torch.count_nonzero(discarded)
        sender.register_forward_hook(observe)
    register(net.input_sender,0)
    for stage,blocks in enumerate(net.stages):
        for block in blocks:
            register(block.sender1,stage)
            register(block.sender2,stage)
    stream=fit.loader(rows,cfg,0,303)[1]
    for i,sample in enumerate(stream):
        row=rows[i]
        phase=i%5
        assert sample['meta']['frame']==row['frame'] and (phase==0)==row['reset_before']
        if phase==0:
            state=None
            buckets.zero_()
        ego=fit.to_device(sample['ego'],torch.device('cuda'))
        _,state,_=net(fit.spatial_input(source,ego),source.core,state,ordinary=row['window_ordinary'])
        if phase==4:
            records.append(dict(scene=row['scene'],frames=[r['frame'] for r in rows[i-4:i+1]],totals=buckets.cpu().tolist()))
    assert len(records)*5==len(rows)
    with out.open('x') as f:
        json.dump(dict(domain=domain,arm=arm,parent=fit.digest(ck),manifest=fit.digest(manifest),
                       total_train_scenes=len(all_scenes),columns=columns,records=records,
                       scope='First complete five-scan training window per eligible scene. Each sender sees identical actual carry-network targets in its two modes. No AP, traffic matching or timing.'),f)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('datasets',type=Path)
    p.add_argument('domain',choices=('v2v','dair'))
    p.add_argument('arm',choices=('native','qat'))
    p.add_argument('out',type=Path)
    args=p.parse_args()
    run(args.root,args.datasets,args.domain,args.arm,args.out)
