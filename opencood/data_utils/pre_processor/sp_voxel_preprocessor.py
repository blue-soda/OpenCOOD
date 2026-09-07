# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class NumpyVoxelGenerator(object):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_num_points = int(max_num_points)
        self.max_voxels = int(max_voxels)
        grid = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.round(grid).astype(np.int64)

    def generate(self, points):
        points = np.asarray(points, dtype=np.float32)
        xyz = points[:, :3]
        mask = np.all((xyz >= self.point_cloud_range[:3]) & (xyz < self.point_cloud_range[3:6]), axis=1)
        points = points[mask]
        if points.shape[0] == 0:
            return (
                np.zeros((0, self.max_num_points, points.shape[1] if points.ndim == 2 else 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        voxel_xyz = np.floor((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        voxel_xyz = np.minimum(voxel_xyz, self.grid_size.astype(np.int32) - 1)

        voxel_map = {}
        voxel_features = []
        voxel_coords = []
        voxel_num_points = []
        feature_dim = points.shape[1]

        for point, coord_xyz in zip(points, voxel_xyz):
            coord_zyx = (int(coord_xyz[2]), int(coord_xyz[1]), int(coord_xyz[0]))
            voxel_idx = voxel_map.get(coord_zyx)
            if voxel_idx is None:
                if len(voxel_features) >= self.max_voxels:
                    continue
                voxel_idx = len(voxel_features)
                voxel_map[coord_zyx] = voxel_idx
                voxel_features.append(np.zeros((self.max_num_points, feature_dim), dtype=np.float32))
                voxel_coords.append(coord_zyx)
                voxel_num_points.append(0)

            num = voxel_num_points[voxel_idx]
            if num < self.max_num_points:
                voxel_features[voxel_idx][num] = point
                voxel_num_points[voxel_idx] = num + 1

        return (
            np.asarray(voxel_features, dtype=np.float32),
            np.asarray(voxel_coords, dtype=np.int32),
            np.asarray(voxel_num_points, dtype=np.int32),
        )


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        voxel_generator_cls = None
        self.spconv_version = 1
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            voxel_generator_cls = VoxelGenerator
        except Exception:
            try:
                from spconv.utils import VoxelGenerator
                voxel_generator_cls = VoxelGenerator
            except Exception:
                voxel_generator_cls = NumpyVoxelGenerator
                self.spconv_version = 0

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']
        
        # whether there are more than on frames in the past
        if 'past_k' in preprocess_params and preprocess_params['past_k'] > 0:
            self.sweep = True
        else:
            self.sweep = False

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        self.voxel_generator = voxel_generator_cls(
            voxel_size=self.voxel_size,
            point_cloud_range=self.lidar_range,
            max_num_points=self.max_points_per_voxel,
            max_voxels=self.max_voxels
        )

    def preprocess(self, pcd_np):
        data_dict = {}
        voxel_output = self.voxel_generator.generate(pcd_np)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points


        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
