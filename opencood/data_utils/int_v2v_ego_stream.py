"""Single-agent V2V4Real scans using the native OpenCOOD color and local-box conventions."""
import json
from pathlib import Path
import re

import numpy as np
import yaml

from opencood.data_utils.int_ego_stream import DairEgoStreamDataset, sha256
from opencood.utils import pcd_utils


def ego_cars(path):
    text = Path(path).read_text()
    section = re.search(r'^vehicles:.*?(?=^\w+:|\Z)', text, re.M | re.S)
    if section is None:
        raise ValueError(('missing vehicles', path))
    # Pose metadata uses serialized NumPy objects. Same-sensor boxes need only this safe mapping.
    vehicles = yaml.safe_load(section.group())['vehicles'] or {}
    return {k:v for k,v in vehicles.items() if v['obj_type'].lower() == 'car'}


def v2v_rows(manifest, split, datasets=None):
    assert sha256(manifest) == '29cad216b5f3af49f727b4781473afd5ee6ee8993e773e2cc3fbcf1b90866fa4'
    rows = []
    last_scene, last_id, segment = None, None, 0
    for index, row in enumerate(map(json.loads,Path(manifest).read_text().splitlines())):
        if row['split'] != split:
            continue
        if datasets is not None:
            for field in ('pcd','annotation'):
                row[field] = str(Path(datasets) / Path(row[field]).relative_to('/data2/cyk/gzc/dataset'))
        raw_id = int(row['frame'])
        reset = last_scene != row['scene'] or raw_id != last_id + 1
        if reset:
            segment += 1
        assert Path(row['pcd']).parent.name == Path(row['annotation']).parent.name == '0'
        rows.append(dict(row, source_frame=row['frame'], frame=str(index), dataset='v2v_single',
                         label=row['annotation'], segment='%s:%d' % (row['scene'],segment),
                         supervised=True, reset_before=reset))
        last_scene,last_id = row['scene'],raw_id
    return rows


class V2VEgoStreamDataset(DairEgoStreamDataset):
    def read_points(self, row):
        points = pcd_utils.pcd_to_np(row['pcd'])
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError(('invalid single-car point data',row['pcd']))
        return points

    def object_centers(self, row):
        objects = {}
        local = [0.]*6
        self.post.project_local_objects(ego_cars(row['label']),objects,local,local,
                                        self.post.params['anchor_args']['cav_lidar_range'])
        ids = list(objects)
        boxes = np.zeros((self.post.params['max_num'],7))
        if len(ids) > len(boxes):
            raise ValueError(('more targets than the configured maximum',row['frame'],len(ids)))
        for i,key in enumerate(ids):
            boxes[i] = objects[key][0]
        return boxes, (np.arange(len(boxes)) < len(ids)).astype(float), ids
