"""Transformation helpers used by the prefixed CoST port.

CoST was developed against an older OpenCOOD utility module which exposed
``align_features``.  Keeping this implementation isolated avoids changing
the behavior of the repository-wide transformation utilities.
"""

import torch
import torch.nn.functional as F


def warp_affine_simple(src, matrix, dsize, align_corners=False):
    """Warp a BCHW tensor with a batch of 2-D affine matrices."""
    batch, channels, _, _ = src.size()
    grid = F.affine_grid(
        matrix,
        [batch, channels, dsize[0], dsize[1]],
        align_corners=align_corners,
    ).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


def _pose_to_tfm(pose):
    """Convert [x, y, yaw] or [x, y, z, roll, yaw, pitch] to matrices."""
    if pose.shape[1] == 4 and pose.shape[2] == 4:
        return pose
    if pose.shape[1] == 3:
        x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
        tfm = torch.eye(4, device=pose.device, dtype=pose.dtype).view(1, 4, 4)
        tfm = tfm.repeat(pose.shape[0], 1, 1)
        tfm[:, 0, 0] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 1] = -torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 0] = torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 1] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 3], tfm[:, 1, 3] = x, y
        return tfm
    raise ValueError(f"Unsupported pose shape: {tuple(pose.shape)}")


def get_pairwise_transformation_torch(lidar_poses, max_cav):
    """Return pairwise transforms with the CoST/OpenCOOD convention."""
    batch = lidar_poses.shape[0]
    pairwise = torch.eye(4, device=lidar_poses.device, dtype=lidar_poses.dtype)
    pairwise = pairwise.view(1, 1, 1, 4, 4).repeat(
        batch, max_cav, max_cav, 1, 1
    )
    for b in range(batch):
        transforms = lidar_poses[b]
        if transforms.shape[-1] != 4:
            transforms = _pose_to_tfm(transforms)
        for i in range(len(transforms)):
            for j in range(len(transforms)):
                if i != j:
                    pairwise[b, i, j] = torch.linalg.solve(
                        transforms[j], transforms[i]
                    )
    return pairwise


def align_features(features, ego_poses, _d=1.6, target_pos=-1):
    """Align temporal BEV features to the selected ego pose."""
    if target_pos not in (-1, 0):
        raise ValueError("target_pos must be -1 or 0")
    batch, frames, _, height, width = features.shape
    pairwise = get_pairwise_transformation_torch(ego_poses, max_cav=frames)
    pairwise = pairwise[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    pairwise[..., 0, 1] *= height / width
    pairwise[..., 1, 0] *= width / height
    pairwise[..., 0, 2] /= (_d * width) / 2
    pairwise[..., 1, 2] /= (_d * height) / 2
    aligned = []
    for b in range(batch):
        aligned.append(
            warp_affine_simple(
                features[b], pairwise[b, target_pos, :frames], (height, width)
            )
        )
    return torch.stack(aligned)
