"""DAIR vehicle-only causal stream. Never load infrastructure observations/GT."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils import pcd_utils
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def build_stream_manifest(root, max_gap_s=.25, max_speed_mps=80.):
    """Assign all scans by existing scene splits; labels only from split indices.

    A missing/empty file forces a new segment. Geometry and time discontinuities
    force new segments too. Unassigned scenes are not silently added to training.
    """
    root = Path(root)
    split_ids = {s: set(read_json(root / (s + '.json'))) for s in ['train', 'val']}
    infos = read_json(root / 'vehicle-side/data_info.json')
    by_id = {Path(x['pointcloud_path']).stem: x for x in infos}
    scene_splits = defaultdict(set)
    for split, ids in split_ids.items():
        for fid in ids:
            scene_splits[str(by_id[fid]['batch_id'])].add(split)
    mixed = {k: sorted(v) for k, v in scene_splits.items() if len(v) > 1}
    if mixed:
        raise ValueError('Train/val share scenes: %s' % mixed)
    groups = defaultdict(list)
    for fid, info in by_id.items():
        scene = str(info['batch_id'])
        if scene in scene_splits:
            groups[scene].append((int(info['pointcloud_timestamp']), fid, info))
    result = dict(train=[], val=[], excluded=[], boundaries=[])
    for scene in sorted(groups):
        split = next(iter(scene_splits[scene]))
        previous, segment = None, 0
        for stamp, fid, info in sorted(groups[scene]):
            pcd = root / 'vehicle-side' / info['pointcloud_path']
            if not pcd.is_file() or pcd.stat().st_size == 0:
                result['excluded'].append(dict(frame=fid, split=split, reason='missing_or_zero_byte_pcd'))
                previous = None
                segment += 1
                continue
            lidar_calib = root / 'vehicle-side' / info['calib_lidar_to_novatel_path']
            world_calib = root / 'vehicle-side' / info['calib_novatel_to_world_path']
            pose = veh_side_rot_and_trans_to_trasnformation_matrix(read_json(lidar_calib), read_json(world_calib))
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
                raise ValueError('Invalid pose: %s' % fid)
            if previous is not None:
                dt = (stamp - previous['timestamp_us']) / 1e6
                distance = np.linalg.norm(pose[:2, 3] - np.asarray(previous['pose'])[:2, 3])
                reason = None
                if dt <= 0 or dt > max_gap_s:
                    reason = 'time_gap'
                elif distance > max_speed_mps * dt + .5:
                    reason = 'pose_jump'
                if reason:
                    segment += 1
                    result['boundaries'].append(dict(frame=fid, previous=previous['frame'],
                                                     reason=reason, delta_t=dt, distance_m=float(distance)))
            supervised = fid in split_ids[split]
            label = root / 'vehicle-side' / info['label_lidar_std_path']
            if supervised and not label.is_file():
                raise FileNotFoundError(str(label))
            row = dict(frame=fid, scene=scene, segment='%s:%04d' % (scene, segment),
                       timestamp_us=stamp, pose=pose.tolist(), pcd=str(pcd),
                       label=str(label) if supervised else None, supervised=supervised,
                       split=split, reset_before=False)
            result[split].append(row)
            previous = row
    result['protocol'] = dict(max_gap_s=max_gap_s, max_speed_mps=max_speed_mps,
                              pose_source='dataset LiDAR->Novatel->world (reference localization)',
                              labels='vehicle-side only, original split IDs',
                              unlabelled='same-split scenes only, update state without loss or evaluation',
                              augmentation='disabled for initial controlled port',
                              hashes={s+'.json': sha256(root/(s+'.json')) for s in ['train', 'val']})
    result['summary'] = {s: dict(scans=len(result[s]), labels=sum(r['supervised'] for r in result[s]),
                               scenes=len(set(r['scene'] for r in result[s])),
                               segments=len(set(r['segment'] for r in result[s]))) for s in ['train', 'val']}
    return result


def training_order(rows, length=4, seed=303):
    if length < 1:
        raise ValueError('Clip length must be positive')
    groups = defaultdict(list)
    for row in rows:
        groups[row['segment']].append(row)
    clips = []
    for group in groups.values():
        for start in range(0, len(group), length):
            clip = copy.deepcopy(group[start:start+length])
            if any(row['supervised'] for row in clip):
                clip[0]['reset_before'] = True
                clips.append(clip)
    random.Random(seed).shuffle(clips)
    return [row for clip in clips for row in clip]


class DairEgoStreamDataset(Dataset):
    def __init__(self, rows, config, seed=303):
        self.rows, self.config, self.seed = rows, config, seed
        # Same point/voxel caps for training and eval in this integration run.
        self.pre = build_preprocessor(config['preprocess'], train=False)
        self.post = build_postprocessor(config['postprocess'], train=False)
        self.anchors = self.post.generate_anchor_box()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        meta = dict(self.rows[index])
        points, _ = pcd_utils.read_pcd(meta['pcd'])
        if len(points) == 0:
            meta['invalid'] = 'decoded_empty_pcd'
            return dict(meta=meta, ego=None)
        rng = np.random.RandomState(self.seed + int(meta['frame']))
        points = points[rng.permutation(len(points))]
        points = pcd_utils.mask_points_by_range(points, self.config['preprocess']['cav_lidar_range'])
        points = pcd_utils.mask_ego_points(points)
        processed = self.pre.collate_batch([self.pre.preprocess(points)])
        processed['batch_size'] = 1
        meta['point_count_after_crop'] = len(points)
        ego = dict(processed_lidar=processed, batch_size=1,
                   anchor_box=torch.from_numpy(self.anchors),
                   transformation_matrix=torch.eye(4), transformation_matrix_clean=torch.eye(4))
        if meta['supervised']:
            annotations = read_json(meta['label'])
            boxes, mask, ids = self.post.generate_object_center_dairv2x_single(
                [{'params': {'vehicles': annotations}}])
            labels = self.post.generate_label(gt_box_center=boxes, anchors=self.anchors, mask=mask)
            ego.update(object_bbx_center=torch.from_numpy(boxes[None]).float(),
                       object_bbx_mask=torch.from_numpy(mask[None]).float(), object_ids=ids,
                       label_dict=self.post.collate_batch([labels]))
        return dict(meta=meta, ego=ego)


def stream_collate(batch):
    if len(batch) != 1:
        raise ValueError('This runner requires batch_size=1 and explicit scene state')
    return batch[0]
