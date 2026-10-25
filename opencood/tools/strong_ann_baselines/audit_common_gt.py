"""Export our exact paired-dataset GT without loading point clouds; verify first two via dataset."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
base=Path('/data1/zhh/gzc');repo=base/'OpenCOOD';sys.path.insert(0,str(repo))
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets.paired_dair_fusion import PairedDAIRFusion,load_pair_geometry
from opencood.utils.transformation_utils import tfm_to_pose
out=base/'diagnostics/ann_baselines_20261025';data=base/'dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure'
cfg=yaml_utils.load_yaml(str(repo/'opencood/hypes_yaml/dair-v2x/snn/e3dsnn_pair_count4_residual_181.yaml'))
cfg.update(data_dir=str(data),root_dir=str(data/'train.json'),validate_dir=str(data/'val.json'),test_dir=str(data/'val.json'),data_manifest=str(base/'diagnostics/e3dsnn_event_utility_source_20261022/pair_manifest.json'))
post=build_postprocessor(cfg['postprocess'],train=False)
dataset=PairedDAIRFusion(cfg,visualize=False,train=False)
ids=json.loads((out/'common_val.json').read_text());assert dataset.data==ids
target=out/'common_gt';target.mkdir(exist_ok=True)
rows=[]
for i,fid in enumerate(ids):
    info=dataset.co_idx2info[fid];pose,_,_=load_pair_geometry(data,info)
    labels=json.loads((data/info['cooperative_label_path']).read_text());labels=[o for o in labels if o['type'] in ('Car','Van','Bus','Truck')]
    centers,mask,object_ids=post.generate_object_center_dairv2x([{'params':{'vehicles':labels}}],tfm_to_pose(pose))
    content={'object_bbx_center':torch.from_numpy(centers),'object_bbx_mask':torch.from_numpy(mask),'object_ids':object_ids,'transformation_matrix_clean':torch.eye(4)}
    gt=post.generate_gt_bbx({'ego':content}).numpy()
    if i<2:
        actual=dataset.collate_batch_test([dataset[i]])
        original=post.generate_gt_bbx(actual).numpy()
        assert np.array_equal(gt,original),'Fast GT export differs from actual dataset'
    p=target/(fid+'.npy');np.save(str(p),gt)
    rows.append({'frame_id':fid,'gt_count':len(gt),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report={'samples':len(rows),'gt_boxes':sum(x['gt_count'] for x in rows),'verified_first_two_against_actual_dataset':True,'frames':rows,'config':cfg['postprocess']}
(out/'common_gt_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('frames','config')}))
