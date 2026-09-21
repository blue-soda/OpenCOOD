"""Frozen single detector -> geometry-bearing messages, without target labels."""
import torch
import torch.nn.functional as F
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle


def diverse_indices(xyz, count):
    if len(xyz) <= count:
        return torch.arange(len(xyz), device=xyz.device)
    # Bound FPS work before selecting spatially distributed keypoints.
    pool = torch.linspace(0, len(xyz) - 1, min(len(xyz), 2048), device=xyz.device).long()
    if len(pool) <= count:
        return pool
    points = xyz[pool, :2]
    distance = points.new_full((len(points),), float('inf'))
    selected = []
    index = points.square().sum(1).argmax()
    for _ in range(count):
        selected.append(index)
        distance = torch.minimum(distance, (points - points[index]).square().sum(1))
        index = distance.argmax()
    return pool[torch.stack(selected)]


class CapriSender(PointPillarCodyntrustSingle):
    def __init__(self, args, settings):
        super().__init__(args)
        self.settings = settings
        self.lidar_range = args['lidar_range']
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def messages(self, processed, frame_count, anchors, postprocessor):
        raw = processed['voxel_features']
        coords = processed['voxel_coords']
        counts = processed['voxel_num_points']
        # Actual retained points, not voxel centroids or BEV cell centers.
        xyz = raw[:, 0, :3].clone()
        batch = {key: processed[key] for key in ('voxel_features', 'voxel_coords', 'voxel_num_points')}
        if len(coords):
            batch = self.scatter(self.pillar_vfe(batch))
            spatial = batch['spatial_features']
            if len(spatial) < frame_count:
                spatial = torch.cat((spatial, spatial.new_zeros(frame_count-len(spatial), *spatial.shape[1:])))
        else:
            spatial = raw.new_zeros(frame_count, 64, self.scatter.ny, self.scatter.nx)
        features = self.backbone({'spatial_features': spatial})['spatial_features_2d']
        if self.shrink_flag:
            features = self.shrink_conv(features)
        outputs = {'psm_single': self.cls_head(features), 'rm_single': self.reg_head(features)}
        if self.use_dir:
            outputs['dm_single'] = self.dir_head(features)
        predictions = postprocessor.single_post_process(
            outputs, torch.eye(4, device=raw.device).repeat(frame_count, 1, 1),
            torch.zeros(frame_count, device=raw.device), anchors,
            num_sweeps=frame_count, num_roi_thres=-1)
        result = []
        for index in range(frame_count):
            pred = predictions[index]
            boxes, scores = pred['pred_box_center_tensor'], pred['scores']
            selected = scores.argsort(descending=True)[:self.settings['max_instances']]
            boxes, scores = boxes[selected], scores[selected]
            points = xyz[coords[:, 0] == index]
            density = counts[coords[:, 0] == index].to(raw.dtype)
            if len(points):
                grid = points[:, :2].clone()
                for axis in range(2):
                    grid[:, axis] = 2 * (grid[:, axis]-self.lidar_range[axis]) / (self.lidar_range[axis+3]-self.lidar_range[axis])-1
                desc = F.grid_sample(features[index:index+1], grid.view(1, 1, -1, 2), align_corners=False)[0, :, 0].T
                delta = points[None, :, :3] - boxes[:, None, :3]
                c, s = boxes[:, 6].cos()[:, None], boxes[:, 6].sin()[:, None]
                x = c*delta[..., 0]+s*delta[..., 1]
                y = -s*delta[..., 0]+c*delta[..., 1]
                inside = (x.abs() <= boxes[:, 5:6]/2) & (y.abs() <= boxes[:, 4:5]/2) & (delta[..., 2].abs() <= boxes[:, 3:4]/2+.2)
                expanded = (x.abs() <= boxes[:, 5:6]/2+1) & (y.abs() <= boxes[:, 4:5]/2+1)
                background = ~expanded.any(0) & (points[:, 2] > self.settings['background_min_z'])
            else:
                desc = features.new_zeros(0, features.shape[1])
                inside = torch.zeros(len(boxes), 0, dtype=torch.bool, device=raw.device)
                background = torch.zeros(0, dtype=torch.bool, device=raw.device)
            num = self.settings['points_per_instance']
            inst_xyz = raw.new_zeros(len(boxes), num, 3)
            inst_feat = raw.new_zeros(len(boxes), num, features.shape[1])
            valid = torch.zeros(len(boxes), num, dtype=torch.bool, device=raw.device)
            for j in range(len(boxes)):
                ids = torch.where(inside[j])[0]
                ids = ids[diverse_indices(points[ids], num)]
                inst_xyz[j, :len(ids)] = points[ids]
                inst_feat[j, :len(ids)] = desc[ids]
                valid[j, :len(ids)] = True
            bg_ids = torch.where(background)[0]
            bg_ids = bg_ids[diverse_indices(points[bg_ids], self.settings['background_points'])]
            result.append({'boxes': boxes, 'scores': scores, 'points': inst_xyz,
                           'features': inst_feat, 'valid': valid,
                           'background': {'xyz': points[bg_ids],
                                          'descriptor': F.normalize(desc[bg_ids], dim=-1),
                                          'quality': density[bg_ids].clamp(max=8)/8}})
        return result
