"""Physical receptive-field centers and receiver-side SE(3) resampling."""
import torch
import torch.nn.functional as F


def encoded_grid(voxel_size, lidar_range):
    # Three k=3,s=2,p=1 layers: jump=8, center offset unchanged.
    # Final k=(3,1,1),s=(2,1,1),p=0: z origin += 8 input voxels.
    origin = [lidar_range[i] + voxel_size[i] * offset
              for i, offset in enumerate((.5, .5, 8.5))]
    step = [voxel_size[i] * stride for i, stride in enumerate((8, 8, 16))]
    return origin, step  # xyz, not zyx


def warp_volume(source, source_to_ego, origin_xyz, step_xyz):
    """Pull source [B,C,D,H,W] into ego; returns values and domain coverage.

    Both grids use the declared local physical origin/step. The transform
    includes any sender-side vertical recentering and coherent augmentation.
    Coverage means grid support, not observed free space or visibility.
    """
    b, _, d, h, w = source.shape
    if source_to_ego.shape != (b, 4, 4) or not torch.isfinite(source_to_ego).all():
        raise ValueError('Expected finite [B,4,4] sender-to-ego transforms')
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=source.dtype, device=source.device)
                              for n in (d, h, w)))
    indices = torch.stack((x, y, z), -1)
    origin = source.new_tensor(origin_xyz)
    step = source.new_tensor(step_xyz)
    if (step <= 0).any():
        raise ValueError('Grid step must be positive')
    xyz = indices * step + origin
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[..., :1])), -1).reshape(-1, 4)
    inverse = torch.inverse(source_to_ego.to(source.dtype))
    local = torch.matmul(homogeneous[None], inverse.transpose(1, 2))[..., :3]
    q = (local - origin) / step
    grid = (2 * (q + .5) / source.new_tensor([w, h, d]) - 1).reshape(b, d, h, w, 3)
    result = F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    coverage = F.grid_sample(torch.ones_like(source[:, :1]), grid,
                             mode='bilinear', padding_mode='zeros', align_corners=False)
    # Exact identity path avoids interpolation roundoff in no-transform audits.
    identity = (source_to_ego == torch.eye(4, device=source.device, dtype=source_to_ego.dtype)).all(-1).all(-1)
    result = torch.where(identity[:, None, None, None, None], source, result)
    coverage = torch.where(identity[:, None, None, None, None], torch.ones_like(coverage), coverage)
    return result, coverage
