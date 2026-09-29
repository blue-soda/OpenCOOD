"""Vehicle-only, current-frame DAIR baseline with vehicle-local annotations.

No random infrastructure selection, history eligibility filter or fusion.
"""
from collections import OrderedDict
from pathlib import Path
import json
import hashlib
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR
from opencood.utils import pcd_utils


class SingleDAIRVehicle(LateFusionDatasetDAIR):
    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        manifest_path = params.get('data_manifest')
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text())
            split = manifest['splits']['train' if train else 'val']
            source = Path(params['root_dir'] if train else params['validate_dir'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != split['source_sha256']:
                raise ValueError('Dataset split differs from the audited source')
            if manifest['data_root'] != self.root_dir or manifest['lidar_range'] != params['preprocess']['cav_lidar_range']:
                raise ValueError('Data root or range differs from the audit')
            valid = set(split['valid_ids'])
            self.data = [frame_id for frame_id in self.data if frame_id in valid]
            if len(self.data) != split['valid_count']:
                raise ValueError('Audited sample list does not match reader')
            print('Explicit vehicle PCD audit: {}/{} samples, manifest={}'.format(
                len(self.data), split['original_count'], manifest_path), flush=True)

    def is_valid_id(self, veh_frame_id):
        return veh_frame_id in self.co_idx2info

    def retrieve_base_data(self, idx):
        frame_id = self.data[idx]
        root = Path(self.root_dir)
        info = self.co_idx2info[frame_id]
        cloud, _ = pcd_utils.read_pcd(str(root / info['vehicle_pointcloud_path']))
        if len(cloud) == 0:
            raise ValueError('Empty vehicle PCD: {}'.format(frame_id))
        with (root / 'vehicle-side/label/lidar' / (frame_id + '.json')).open() as stream:
            labels = json.load(stream)
        # Both points and these labels are vehicle-local. No other agent is
        # present, so identity pose avoids unneeded global calibration IO.
        return OrderedDict([(0, {'ego': True, 'lidar_np': cloud,
            'params': {'vehicles': labels, 'lidar_pose': [0., 0., 0., 0., 0., 0.]}})])
