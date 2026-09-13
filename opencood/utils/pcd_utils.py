# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Utility functions related to point cloud
"""

import open3d as o3d
import numpy as np
from pypcd import pypcd
import struct
import os

_PCD_ISSUE_LOG_LIMIT = int(os.environ.get('OPENCOOD_PCD_ISSUE_LOG_LIMIT', 8))
_PCD_ISSUE_REASON_LOG_LIMIT = int(
    os.environ.get('OPENCOOD_PCD_ISSUE_REASON_LOG_LIMIT', 4))
_pcd_issue_log_count = 0
_pcd_issue_reason_counts = {}


def _empty_pcd_points():
    return np.zeros((0, 4), dtype=np.float32)


def _log_pcd_read_issue(pcd_path, reason, exc=None):
    global _pcd_issue_log_count
    if _pcd_issue_log_count >= _PCD_ISSUE_LOG_LIMIT:
        return
    reason_count = _pcd_issue_reason_counts.get(reason, 0)
    if reason_count >= _PCD_ISSUE_REASON_LOG_LIMIT:
        return
    _pcd_issue_log_count += 1
    _pcd_issue_reason_counts[reason] = reason_count + 1
    detail = ''
    if exc is not None:
        detail = ' | %s: %s' % (type(exc).__name__, exc)
    print('[pcd_utils] %s (%d/%d): %s%s' % (
        reason,
        _pcd_issue_log_count,
        _PCD_ISSUE_LOG_LIMIT,
        pcd_path,
        detail))


def _log_pcd_binary_fallback_failure(pcd_path, exc):
    _log_pcd_read_issue(
        pcd_path,
        'binary_compressed parser failed; using Open3D fallback',
        exc)


def _is_corrupt_binary_compressed_error(exc):
    message = str(exc).lower()
    return any(token in message for token in (
        'invalid pcd header',
        'error decompressing',
        'error in compressed data',
        'decompress',
        'unpack requires',
        'buffer',
        'truncated',
    ))

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list

def _pcd_numpy_type(pcd_type, size):
    if pcd_type == 'F':
        return {4: np.float32, 8: np.float64}[size]
    if pcd_type == 'U':
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
    if pcd_type == 'I':
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
    raise ValueError('Unsupported PCD type: %s' % pcd_type)


def _read_pcd_binary_compressed(pcd_path):
    metadata = {}
    with open(pcd_path, 'rb') as pcd_file:
        while True:
            line = pcd_file.readline()
            if not line:
                raise ValueError('Invalid PCD header: %s' % pcd_path)
            line = line.decode('utf-8').strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].lower()
            values = parts[1:]
            if key in ('fields', 'type'):
                metadata[key] = values
            elif key in ('size', 'count'):
                metadata[key] = [int(v) for v in values]
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(values[0])
            elif key == 'data':
                metadata[key] = values[0].lower()
                break

        if metadata.get('data') != 'binary_compressed':
            raise ValueError('Only binary_compressed PCD fallback is supported.')
        if 'count' not in metadata:
            metadata['count'] = [1] * len(metadata['fields'])

        dtype_fields = []
        for field, count, pcd_type, size in zip(
                metadata['fields'],
                metadata['count'],
                metadata['type'],
                metadata['size']):
            np_type = _pcd_numpy_type(pcd_type, size)
            if count == 1:
                dtype_fields.append((field, np_type))
            else:
                for count_idx in range(count):
                    dtype_fields.append(('%s_%04d' % (field, count_idx), np_type))
        dtype = np.dtype(dtype_fields)
        width = metadata.get('points', metadata['width'] * metadata.get('height', 1))

        compressed_size, uncompressed_size = struct.unpack('II', pcd_file.read(8))
        compressed_data = pcd_file.read(compressed_size)
        buf = pypcd.lzf.decompress(compressed_data, uncompressed_size)
        if len(buf) != uncompressed_size:
            raise IOError('Error decompressing PCD data: %s' % pcd_path)

    pc_data = np.zeros(width, dtype=dtype)
    offset = 0
    for name in dtype.names:
        field_bytes = dtype[name].itemsize * width
        pc_data[name] = np.frombuffer(buf[offset:offset + field_bytes],
                                      dtype=dtype[name],
                                      count=width)
        offset += field_bytes
    return pc_data, width


def _read_pcd_open3d_xyz(pcd_path):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    pcd_np_points = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    pcd_np_points[:, :3] = xyz
    return pcd_np_points


def  read_pcd(pcd_path):
    time = None
    try:
        file_size = os.path.getsize(pcd_path)
    except OSError as exc:
        _log_pcd_read_issue(pcd_path, 'pcd file is not readable; using empty cloud', exc)
        return _empty_pcd_points(), time
    if file_size == 0:
        _log_pcd_read_issue(pcd_path, 'pcd file is empty; using empty cloud')
        return _empty_pcd_points(), time

    try:
        pcd = pypcd.PointCloud.from_path(pcd_path)
        pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
        pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
        pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
        pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
        pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    except TypeError:
        # Some DAIR binary_compressed PCD files trigger a bytes/str bug in old
        # pypcd under Python 3. Parse that format locally to keep intensity.
        time = None
        try:
            pc_data, point_num = _read_pcd_binary_compressed(pcd_path)
            pcd_np_points = np.zeros((point_num, 4), dtype=np.float32)
            pcd_np_points[:, 0] = np.transpose(pc_data["x"])
            pcd_np_points[:, 1] = np.transpose(pc_data["y"])
            pcd_np_points[:, 2] = np.transpose(pc_data["z"])
            if "intensity" in pc_data.dtype.names:
                pcd_np_points[:, 3] = np.transpose(pc_data["intensity"]) / 256.0
        except Exception as exc:
            if _is_corrupt_binary_compressed_error(exc):
                _log_pcd_read_issue(
                    pcd_path,
                    'pcd file appears corrupt; using empty cloud',
                    exc)
                return _empty_pcd_points(), time
            _log_pcd_binary_fallback_failure(pcd_path, exc)
            pcd_np_points = _read_pcd_open3d_xyz(pcd_path)
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points, time
