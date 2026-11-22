"""Score the full-width recurrent parent and the sparse local ANN on the locked scene list."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.point_pillar_int import PointPillarINT
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.tools.eval_int_branch_fusion import (
    BranchFusion, CachedResidualPolicy, FloatProjectionAudit, ObservedPolicy,
    compact_group_bucket, digest, loader, local_model, measured_step, prepare_stream,
    read, score, spatial_input, to_device, write_json,
)


class FullWidthParent(PyramidStateDetector):
    def step_spatial(self, spatial, state, meta, requests=None):
        assert requests is None
        return super().step_spatial(spatial, state, meta)


@torch.no_grad()
def run(args):
    inputs = args.long_base / 'inputs'
    source, cfg, *_ = prepare_stream(inputs, args.long_base / 'clips')
    contract = read(args.reference / 'contract.json')
    assert contract['split'] == 'internal' and digest(args.plan) == contract['plan']
    rows = read(args.plan)['rows']['internal']
    reference = [json.loads(line) for line in (args.reference / 'frames.jsonl').read_bytes().splitlines()]
    assert [row['frame'] for row in rows] == [row['frame'] for row in reference]
    assert list(dict.fromkeys(row['scene'] for row in rows)) == contract['scenes']
    members = read(args.plan)['members']
    for row in rows:
        for key in ('pcd','label'):
            if row[key]:
                assert digest(Path(row[key])) == members[row[key]]
    dense = args.arm == 'dense_parent'
    if dense:
        model = FullWidthParent(PointPillarINT(cfg['model']['args'], 'gru').eval(), cfg)
        model.load_state_dict(torch.load(inputs / 'parent.pth', map_location='cpu')['model'], strict=True)
        model.eval().requires_grad_(False).cuda()
    else:
        fusion_block = source.core.readout.layers[0].double_conv[0]
        fusion = BranchFusion(fusion_block[0]).eval()
        fusion_block[0] = fusion
        model = local_model(source, indexed=True)
        projection = FloatProjectionAudit(source.core, live=True, verify=True)
    write_json(args.output/'contract.json',dict(arm=args.arm, parent=contract,
        trace=digest(args.reference/'frames.jsonl'), source=digest(inputs/'parent.pth'),
        code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),training=False))
    ds, stream = loader(rows, cfg, 0, 303)
    state = None
    selector = None
    with (args.output/'frames.jsonl').open('x', buffering=1) as out:
        for i,(sample,ref) in enumerate(zip(stream,reference)):
            meta=sample['meta']
            assert (meta['frame'],meta['scene'])==(ref['frame'],ref['scene'])
            if i==0 or rows[i]['segment']!=rows[i-1]['segment']:
                state=None
                if not dense:
                    selector=ObservedPolicy(model,partial(CachedResidualPolicy,compiler=compact_group_bucket))
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(model if dense else source,ego)
            if not dense:
                reason=model.refresh_reason(state,meta)
                assert reason==ref['refresh']
                fusion.begin(state,reason)
                projection.begin()
                selector.observation=None
            (pred,state,info),ops=measured_step(model,spatial,state,meta,selector)
            record=dict(frame=meta['frame'],scene=meta['scene'])
            if dense:
                assert all(s['actual_write_fraction']==1. for s in info['cache_stages'])
                record['refresh']='dense'
            else:
                assert info['full_refresh']==reason
                ops+=fusion.ledger['float_conv']+2*sum(p['sparse_mac'] for p in projection.ledger.values())
                state['branch_fusion']=fusion.new
                record['refresh']=reason
            record['core_ops']=int(ops)
            if meta['supervised']:
                gt=ds.post.generate_gt_bbx(dict(ego=ego))
                record.update(score(ds.post,ego,pred,gt,record_matches=True))
                assert record['ap'][.7]['gt']==len(ref['points'])
            out.write(json.dumps(record,allow_nan=False)+'\n')
    assert i+1==len(rows)
    write_json(args.output/'report.json',dict(status='COMPLETED',scans=len(rows),optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('arm',choices=('dense_parent','local_ann'))
    for key in ('long-base','plan','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)
