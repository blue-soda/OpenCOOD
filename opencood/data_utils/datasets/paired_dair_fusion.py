"""Audited paired DAIR data, cooperative world labels, vehicle ego output.

Road points remain in a sender-local frame. A calibrated z translation aligns
the grid's vertical datum to the recipient without rotating/projecting XY.
This per-recipient normalization uses calibration only, never GT.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR
from opencood.utils import pcd_utils, box_utils
from opencood.utils.transformation_utils import (
    veh_side_rot_and_trans_to_trasnformation_matrix as vehicle_world,
    inf_side_rot_and_trans_to_trasnformation_matrix as road_world, tfm_to_pose)


def load_pair_geometry(root, info):
    read = lambda p: json.loads((root / p).read_text())
    vid = Path(info['vehicle_pointcloud_path']).stem
    iid = Path(info['infrastructure_pointcloud_path']).stem
    v = vehicle_world(read('vehicle-side/calib/lidar_to_novatel/' + vid + '.json'),
                      read('vehicle-side/calib/novatel_to_world/' + vid + '.json'))
    r = road_world(read('infrastructure-side/calib/virtuallidar_to_world/' + iid + '.json'),
                   info['system_error_offset'])
    transform = np.linalg.solve(v, r)
    if not np.isfinite(transform).all() or abs(transform[2, 2]) < .5:
        raise ValueError('Invalid or near-horizontal road vertical axis')
    # q=S*p, so ego=T*inv(S)*q. The z translation in T*inv(S) is zero.
    shift = float(transform[2, 3] / transform[2, 2])
    inverse_shift = np.eye(4)
    inverse_shift[2, 3] = -shift
    return v, transform @ inverse_shift, shift


def joint_augmentation(config, rng=np.random):
    matrix = np.eye(4)
    for item in config:
        if item['NAME'] == 'random_world_flip':
            for axis in item['ALONG_AXIS_LIST']:
                if axis not in ('x', 'y'):
                    raise ValueError('Unsupported flip axis')
                flip = np.eye(4)
                if rng.random() < .5:
                    flip[1 if axis == 'x' else 0, 1 if axis == 'x' else 0] = -1
                matrix = flip @ matrix
        elif item['NAME'] == 'random_world_rotation':
            angle = rng.uniform(*item['WORLD_ROT_ANGLE'])
            rotation = np.eye(4)
            rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            matrix = rotation @ matrix
        elif item['NAME'] == 'random_world_scaling':
            scale = np.eye(4)
            scale[:3, :3] *= rng.uniform(*item['WORLD_SCALE_RANGE'])
            matrix = scale @ matrix
        else:
            raise ValueError('Unsupported paired augmentation: ' + item['NAME'])
    return matrix


class PairedDAIRFusion(LateFusionDatasetDAIR):
    protocol = 'paired vehicle/infrastructure, cooperative world GT projected to vehicle; no injected delay'

    def is_valid_id(self, frame_id):
        return frame_id in self.co_idx2info

    def __init__(self, params, visualize=False, train=True):
        super().__init__(params, visualize, train)
        manifest = json.loads(Path(params['data_manifest']).read_text())
        if manifest.get('protocol') != 'paired_dair_fusion_v1':
            raise ValueError('A paired data audit is required')
        self.manifest = manifest
        split = manifest['splits']['train' if train else 'val']
        source = Path(params['root_dir'] if train else params['validate_dir'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != split['source_sha256']:
            raise ValueError('Split differs from paired audit')
        if manifest['data_root'] != self.root_dir or manifest['lidar_range'] != params['preprocess']['cav_lidar_range']:
            raise ValueError('Range/root differs from paired audit')
        if hashlib.sha256((Path(self.root_dir) / 'cooperative/data_info.json').read_bytes()).hexdigest() != manifest['cooperative_info_sha256']:
            raise ValueError('Pair metadata changed after audit')
        self.data = split['valid_ids']
        self.timestamps = manifest['timestamps_us']

    def __getitem__(self, index):
        fid = self.data[index]
        root = Path(self.root_dir)
        info = self.co_idx2info[fid]
        vehicle_pose, transform, shift = load_pair_geometry(root, info)
        clouds = [pcd_utils.read_pcd(str(root / info[key]))[0]
                  for key in ('vehicle_pointcloud_path', 'infrastructure_pointcloud_path')]
        if any(len(p) == 0 or not np.isfinite(p).all() for p in clouds):
            raise ValueError('Invalid audited point cloud: ' + fid)
        clouds[1][:, 2] += shift
        labels = json.loads((root / info['cooperative_label_path']).read_text())
        labels = [o for o in labels if o['type'] in ('Car', 'Van', 'Bus', 'Truck')]
        centers, mask, ids = self.post_processor.generate_object_center_dairv2x(
            [{'params': {'vehicles': labels}}], tfm_to_pose(vehicle_pose))
        a = joint_augmentation(self.params['data_augment']) if self.train else np.eye(4)
        transform = a @ transform @ np.linalg.inv(a)
        for p in clouds:
            p[:, :3] = p[:, :3] @ a[:3, :3].T
        valid = mask.astype(bool)
        centers[valid, :3] = centers[valid, :3] @ a[:3, :3].T
        centers[valid, 3:6] *= np.linalg.norm(a[:3, 0])
        angles = centers[valid, 6]
        directions = np.stack((np.cos(angles), np.sin(angles)), 1) @ a[:2, :2].T
        centers[valid, 6] = np.arctan2(directions[:, 1], directions[:, 0])
        region = self.params['preprocess']['cav_lidar_range']
        processed = []
        for i, p in enumerate(clouds):
            p = pcd_utils.mask_points_by_range(p, region)
            if i == 0:
                p = pcd_utils.mask_ego_points(p)
            if not len(p):
                raise ValueError('Empty paired view after crop/augmentation: ' + fid)
            if self.train:
                p = pcd_utils.shuffle_points(p)
            processed.append(self.pre_processor.preprocess(p))
        anchor = self.post_processor.generate_anchor_box()
        result = {'object_bbx_center': centers, 'object_bbx_mask': mask, 'object_ids': ids,
            'anchor_box': anchor, 'processed_lidar': processed[0], 'road_lidar': processed[1],
            'label_dict': self.post_processor.generate_label(gt_box_center=centers, anchors=anchor, mask=mask),
            'road_to_ego': transform.astype(np.float32), 'frame_id': fid,
            'timestamps_us': self.timestamps[fid], 'road_z_shift': shift}
        return {'ego': result}

    def collate_batch_train(self, batch):
        result = super().collate_batch_train(batch)
        rows = [item['ego'] for item in batch]
        result['ego'].update({
            'road_lidar': self.pre_processor.collate_batch([r['road_lidar'] for r in rows]),
            'road_to_ego': torch.from_numpy(np.stack([r['road_to_ego'] for r in rows])),
            'frame_ids': [r['frame_id'] for r in rows],
            'timestamps_us': torch.tensor([r['timestamps_us'] for r in rows], dtype=torch.int64),
            'object_ids': rows[0]['object_ids'],
            'transformation_matrix': torch.eye(4), 'transformation_matrix_clean': torch.eye(4)})
        return result

    def collate_batch_test(self, batch):
        if len(batch) != 1:
            raise ValueError('Evaluation uses batch=1')
        return self.collate_batch_train(batch)
