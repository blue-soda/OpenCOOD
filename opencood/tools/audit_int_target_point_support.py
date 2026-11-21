"""Count input returns inside official DAIR development GT boxes without model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.data_utils.post_processor import build_postprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.check_int_deep_spike_stream import read
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import write_json
from opencood.utils import box_utils, pcd_utils


def run(a):
    cfg=load_yaml(str(a.inputs/'config.yaml'))
    post=build_postprocessor(cfg['postprocess'],train=False)
    manifest=a.long_base/'r328_manifest.json'
    plan_path=a.long_base/'r309_full_plan.json'
    assert digest(plan_path)=='1be45bac9fe7301f91616fa4a7bc2c9c14447ac4053421338d4680ac259e82e6'
    plan=read(plan_path)
    assert digest(manifest)==plan['manifest_sha256'] and digest(a.inputs/'config.yaml')==plan['config_sha256']
    records=list(map(json.loads,a.prior.read_bytes().splitlines()))
    if a.remainder_online:
        contract=read(a.prior.parent/'contract.json')
        assert contract['remainder'] and len(records)==4780 and read(a.prior.parent/'report.json')['status']=='COMPLETED'
        ref={r['frame']:r['arms']['carry'] for r in records if 'ap' in r['arms']['carry']}
        assert len(ref)==1280
    else:
        ref={r['frame']:r for r in records if 'ap' in r}
        assert len(ref)==458
    counts=np.zeros((3,4),dtype=int)
    fields=Counter()
    id_fields=Counter()
    labels=0
    processed=set()
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for row in read(manifest)['val']:
            frame=row['frame']
            if frame not in ref:continue
            paths={}
            for key,folder in [('label','label/lidar'),('pcd','velodyne')]:
                paths[key]=a.vehicle/folder/Path(row[key]).name
                assert digest(paths[key])==plan['input_files'][row[key]]['sha256']
            annotations=read(paths['label'])
            for obj in annotations:
                fields.update(obj.keys())
                id_fields.update(k for k,v in obj.items() if 'id' in k.lower() and v is not None)
            labels+=len(annotations)
            boxes,mask,ids=post.generate_object_center_dairv2x_single([{'params':{'vehicles':annotations}}])
            gt=post.generate_gt_bbx(dict(ego=dict(object_bbx_center=torch.from_numpy(boxes[None]).float(),
                  object_bbx_mask=torch.from_numpy(mask[None]).float(),object_ids=ids,transformation_matrix_clean=torch.eye(4)))).numpy()
            ranges=np.digitize(np.linalg.norm(gt[:,:,:2].mean(1),axis=1),[30.,50.])
            assert np.bincount(ranges,minlength=3).tolist()==ref[frame]['targets']
            xyz,_=pcd_utils.read_pcd(str(paths['pcd']))
            xyz=pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(xyz,cfg['preprocess']['cav_lidar_range']))[:,:3]
            supports=[]
            for box,region in zip(gt,ranges):
                count=len(box_utils.get_points_in_rotated_box_3d(xyz,box))
                supports.append(count)
                counts[region,np.searchsorted([0,5,20],count)]+=1
            log.write(json.dumps(dict(frame=frame,scene=row['scene'],segment=row['segment'],
                      gt_xyz_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),distance_bins=ranges.tolist(),points=supports))+'\n')
            processed.add(frame)
    assert processed==set(ref) and counts.sum()==(17179 if a.remainder_online else 5792)
    write_json(a.output/'report.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               prior=digest(a.prior),manifest=digest(manifest),model_forwards=0,scans=len(processed),gt=int(counts.sum()),
               raw_annotations=labels,fields=fields,non_null_id_fields=id_fields,
               support_by_range=counts.tolist(),bins=['0','1-5','6-20','21+'],range_m=['<30','30-50','>=50']))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','vehicle','long-base','prior','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--remainder-online',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

