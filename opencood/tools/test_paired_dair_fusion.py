"""Paired reader contract using synthetic clouds, calibration and world GT."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.paired_dair_fusion import PairedDAIRFusion, load_pair_geometry, joint_augmentation
from opencood.utils import box_utils


class PairedReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_yaml('opencood/hypes_yaml/dair-v2x/snn/e3dsnn_pair_count4_residual.yaml')
        self.cfg['data_dir'] = str(self.root)
        self.cfg['root_dir'] = self.cfg['validate_dir'] = str(self.root / 'split.json')
        self.cfg['data_manifest'] = str(self.root / 'manifest.json')
        self.cfg['preprocess']['voxel_backend'] = 'numpy'
        self.cfg['data_augment'] = []
        self.write('split.json', ['000001'])
        row = {'vehicle_pointcloud_path': 'vehicle-side/velodyne/000001.pcd',
               'infrastructure_pointcloud_path': 'infrastructure-side/velodyne/000002.pcd',
               'cooperative_label_path': 'cooperative/label_world/000001.json',
               'system_error_offset': {'delta_x': 0, 'delta_y': 0}}
        self.row = row
        self.write('cooperative/data_info.json', [row])
        self.write('infrastructure-side/data_info.json', [{'pointcloud_path': 'velodyne/000002.pcd'}])
        calibration = {'rotation': np.eye(3).tolist(), 'translation': [[0], [0], [0]]}
        self.write('vehicle-side/calib/lidar_to_novatel/000001.json', {'transform': calibration})
        self.write('vehicle-side/calib/novatel_to_world/000001.json', calibration)
        self.write('infrastructure-side/calib/virtuallidar_to_world/000002.json',
                   dict(calibration, translation=[[5], [0], [6]]))
        corners = box_utils.boxes_to_corners_3d(np.asarray([[10., 3., -1., 1.56, 2., 4.5, 0.]]), 'hwl')[0]
        self.write(row['cooperative_label_path'], [{'type': 'Car', 'world_8_points': corners.tolist()}])
        split = {'source_sha256': hashlib.sha256((self.root/'split.json').read_bytes()).hexdigest(),
                 'valid_ids': ['000001'], 'valid_count': 1}
        self.write('manifest.json', {'protocol': 'paired_dair_fusion_v1', 'data_root': str(self.root),
             'lidar_range': self.cfg['preprocess']['cav_lidar_range'],
             'cooperative_info_sha256': hashlib.sha256((self.root/'cooperative/data_info.json').read_bytes()).hexdigest(),
             'splits': {'train': split, 'val': split}, 'timestamps_us': {'000001': [100, 125]}})

    def write(self, path, value):
        p = self.root/path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def test_vertical_recenter_preserves_physical_position(self):
        _, transform, shift = load_pair_geometry(self.root, self.row)
        self.assertEqual(shift, 6)
        q = np.asarray([5., 3., -7., 1.])
        shifted = q.copy(); shifted[2] += shift
        np.testing.assert_allclose(transform @ shifted, [10, 3, -1, 1])

    def test_joint_augmentation_conjugates_transform(self):
        _, t, _ = load_pair_geometry(self.root, self.row)
        a = joint_augmentation([{'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [.4, .4]},
            {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [1.1, 1.1]},
            {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}], np.random.RandomState(0))
        p = np.asarray([5., 2., -1., 1.])
        np.testing.assert_allclose(a @ t @ p, (a @ t @ np.linalg.inv(a)) @ (a @ p))

    def test_world_gt_two_views_collate_and_deterministic_eval(self):
        def cloud(path):
            p = np.asarray([[10., 3., -1., .5], [10.2, 3.1, -1.1, .6]], dtype=np.float32)
            if 'infrastructure-side' in path:
                p[:, 0] -= 5; p[:, 2] -= 6
            return p, None
        dataset = PairedDAIRFusion(self.cfg, train=False)
        with patch('opencood.data_utils.datasets.paired_dair_fusion.pcd_utils.read_pcd', side_effect=cloud):
            sample = dataset[0]
            repeated = dataset[0]
        ego = sample['ego']
        self.assertEqual(int(ego['object_bbx_mask'].sum()), 1)
        np.testing.assert_allclose(ego['object_bbx_center'][0, :3], [10., 3., -1.], atol=1e-5)
        np.testing.assert_array_equal(ego['processed_lidar']['voxel_features'], repeated['ego']['processed_lidar']['voxel_features'])
        batch = dataset.collate_batch_train([sample, repeated])['ego']
        self.assertEqual(tuple(batch['road_to_ego'].shape), (2, 4, 4))
        self.assertEqual(set(batch['road_lidar']['voxel_coords'][:, 0].tolist()), {0, 1})
        gt = dataset.post_processor.generate_gt_bbx_by_iou(dataset.collate_batch_test([sample]))
        self.assertEqual(tuple(gt.shape), (1, 8, 3))


if __name__ == '__main__':
    unittest.main()
