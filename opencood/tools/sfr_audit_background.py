"""Distinguish sparse matching limitations from cross-view background disagreement."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0, os.getcwd())
from opencood.models.sfr.calibration import BackgroundCalibration, weighted_fit
from opencood.models.sfr.geometry import se2, transform_points
from opencood.tools.sfr_run import seed, digest
from opencood.tools.train_utils import to_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=128)
    args=p.parse_args();seed(303)
    cache=torch.load(args.cache,map_location='cpu');saved=torch.load(args.checkpoint,map_location='cpu')
    cfg=saved['config'];model=BackgroundCalibration(cfg['calibration']).cuda().eval()
    model.load_state_dict(saved['model']);rows=cache['pairs']
    result=[]
    indices=torch.linspace(0,len(rows)-1,min(args.samples,len(rows))).long().tolist()
    with torch.no_grad():
        for index in indices:
            row=rows[index];source=to_device(row['source'],'cuda');target=to_device(row['target'],'cuda')
            delta=(torch.rand(3,device='cuda')*2-1)*torch.tensor([1.,1.,.03],device='cuda')
            perturb=se2(delta);expected=torch.inverse(perturb)
            record=dict(index=index,frame=row['meta']['frame_id'],before_m=float(delta[:2].norm()),arms={})
            for name,reference,change in [('cross_nominal',target,False),('cross_perturbed',target,True),('same_perturbed',source,True)]:
                input_source=dict(source,xyz=transform_points(source['xyz'],perturb)) if change else source
                _,raw,stats=model(input_source,reference)
                record['arms'][name]=dict(stats,translation_error_m=float((raw[:2]-(expected[:2,3] if change else 0.)).norm()))
            distance=torch.cdist(source['xyz'][:,:2],target['xyz'][:,:2])
            values,nearest=distance.min(1)
            record['cross_nearest_median_m']=float(values.median())
            record['cross_nearest_height_median_m']=float((source['xyz'][:,2]-target['xyz'][nearest,2]).abs().median())
            record['source_xy_extent']=(source['xyz'][:,:2].max(0).values-source['xyz'][:,:2].min(0).values).tolist()
            result.append(record)
    summary={}
    for name in ('cross_nominal','cross_perturbed','same_perturbed'):
        items=[r['arms'][name] for r in result]
        summary[name]={k:sum(float(v[k]) for v in items if v.get(k) is not None)/max(sum(v.get(k) is not None for v in items),1)
                       for k in ('accepted','overlap','residual_before','residual_after','translation_error_m')}
    output=dict(cache_sha256=digest(args.cache),checkpoint_sha256=digest(args.checkpoint),records=result,summary=summary,
                before_m=sum(r['before_m'] for r in result)/len(result))
    Path(args.output).write_text(json.dumps(output,indent=2));print(json.dumps(dict(summary=summary,before_m=output['before_m'])))


if __name__=='__main__':main()
