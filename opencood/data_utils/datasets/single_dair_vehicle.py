"""Vehicle-only, current-frame DAIR baseline with vehicle-local annotations.

No random infrastructure selection, history eligibility filter or fusion.
"""
from collections import OrderedDict
from pathlib import Path
import json
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR
from opencood.utils import pcd_utils


class SingleDAIRVehicle(LateFusionDatasetDAIR):
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
