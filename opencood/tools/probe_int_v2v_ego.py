"""Frozen, ego-only V2V4Real geometry and inference check on training scans."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opencood.data_utils.int_ego_stream import DairEgoStreamDataset
from opencood.data_utils.int_v2v_ego_stream import ego_cars
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.probe_int_local_ann_interfaces import ReusedProjection
from opencood.tools.select_int_stream_variant import Interface, FusionCost, SparseFusionLayer
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid
from opencood.utils import pcd_utils


@torch.no_grad()
def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    data = DairEgoStreamDataset([], cfg)
    src = fit.build_source(a.root/'inputs', cfg)
    net = RecoveryPyramid(src.core,a.root/'initial.pt','fine_temporal').cuda()
    scales = fit.read(a.root/'scales707/scales.json')
    ck = a.root/'fine_temporal_707.pt'
    assert scales['parent_sha'] == fit.digest(ck) and scales['arm'] == 'delta_medium'
    net.load_state_dict(torch.load(ck,map_location='cpu')['net'])
    net.eval()
    sender = Interface(.25).cuda()
    sender.collect = False
    sender.log_step = nn.Parameter(torch.tensor(scales['log_step'][0],device='cuda'))
    src.core.readout.register_forward_pre_hook(lambda module,args:(sender(args[0]),))
    fusion = FusionCost(True,False,True)
    for parent in list(src.core.readout.modules()):
        for name, conv in list(parent.named_children()):
            if isinstance(conv,nn.Conv2d):
                setattr(parent,name,SparseFusionLayer(conv,fusion))
    projections = []
    for branch in src.core.deblocks:
        branch[0] = ReusedProjection(branch[0])
        projections.append(branch[0])
    rows = [r for r in map(json.loads,a.manifest.read_text().splitlines()) if r['split']=='train']
    stats = {t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)}
    drives, selected = set(), []
    count = targets = raw_points = 0
    for scene, group in itertools.groupby(rows, key=lambda r:r['scene']):
        seq = list(group)
        if seq[0]['drive'] in drives:
            continue
        drives.add(seq[0]['drive'])
        selected.append(scene)
        state, age, prior = None, 0, None
        for row in seq[:25]:
            assert Path(row['pcd']).parent.name == '0' and Path(row['annotation']).parent.name == '0'
            frame = int(row['frame'])
            if prior is None or frame != prior + 1 or age == 5:
                state, age, sender.previous = None,0,None
                fusion.history.clear()
                for p in projections:
                    p.old = p.output = None
            prior = frame
            points = pcd_utils.pcd_to_np(row['pcd'])
            assert points.shape[1] == 4 and len(points) and np.isfinite(points).all()
            raw_points += len(points)
            points = points[np.random.RandomState(707+frame).permutation(len(points))]
            points = pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(points,cfg['preprocess']['cav_lidar_range']))
            lidar = data.pre.collate_batch([data.pre.preprocess(points)])
            lidar['batch_size'] = 1
            local = [0.]*6
            cars = ego_cars(row['annotation'])
            projected = {}
            data.post.project_local_objects(cars, projected, local, local,
                                            data.post.params['anchor_args']['cav_lidar_range'])
            ids = list(projected)
            boxes = np.zeros((data.post.params['max_num'], 7))
            assert len(ids) <= len(boxes)
            mask = np.zeros(len(boxes))
            for i, key in enumerate(ids):
                boxes[i] = projected[key][0]
                mask[i] = 1
            ego = dict(processed_lidar=lidar, batch_size=1,anchor_box=torch.from_numpy(data.anchors),
                       object_bbx_center=torch.from_numpy(boxes[None]).float(),
                       object_bbx_mask=torch.from_numpy(mask[None]).float(),object_ids=ids,
                       transformation_matrix=torch.eye(4),transformation_matrix_clean=torch.eye(4))
            ego = fit.to_device(ego,torch.device('cuda'))
            pred,state,_ = net(fit.spatial_input(src,ego),src.core,state,ordinary=bool(age))
            gt = data.post.generate_gt_bbx(dict(ego=ego))
            frame_stats = fit.score(data.post,ego,pred,gt)['ap']
            for t in stats:
                for key in stats[t]:
                    stats[t][key] += frame_stats[t][key]
            count += 1
            targets += len(gt)
            age += 1
        if len(drives)==2:
            break
    assert count == 50 and targets > 0
    fit.write_json(a.out/'result.json',dict(scans=count,scenes=selected,gt=targets,points=raw_points,
        ap50=fit.calculate_ap(stats,.5)[0],ap70=fit.calculate_ap(stats,.7)[0],source='DAIR seed707',
        split='first 25 paired scans from each of two distinct training drives; fixed ego 0, Car only',
        training_steps=0, dataset_time='ordered frame ids; no wall-clock or motion-based claim'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','out'):
        p.add_argument(name,type=Path)
    run(p.parse_args())
