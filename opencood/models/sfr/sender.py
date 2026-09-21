"""Frozen wide single detector, deterministic observed geometry, candidate-assisted FG."""
import numpy as np
import torch
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle
from .geometry import cell_centers
from .codec import quantize, encode, decode


def geometry_map(voxels, counts, shape, bounds):
    """Occupancy, retained-point log density, mean height, height std.

    Unobserved cells are NOT known free space. Statistics describe retained
    voxel points, not all original returns or ray-traced visibility.
    """
    h, w = shape
    selected = torch.arange(voxels.shape[1], device=voxels.device)[None] < counts[:, None]
    xyz = voxels[:, :, :3][selected]
    x = ((xyz[:, 0]-bounds[0])*w/(bounds[3]-bounds[0])).floor().long()
    y = ((xyz[:, 1]-bounds[1])*h/(bounds[4]-bounds[1])).floor().long()
    keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    index, z = y[keep]*w+x[keep], xyz[keep, 2]
    n = voxels.new_zeros(h*w).index_add(0, index, torch.ones_like(z))
    mean = voxels.new_zeros(h*w).index_add(0, index, z)/n.clamp_min(1)
    second = voxels.new_zeros(h*w).index_add(0, index, z.square())/n.clamp_min(1)
    valid = n > 0
    maps = torch.stack((valid.to(voxels.dtype), n.log1p(), mean, (second-mean.square()).clamp_min(0).sqrt()), -1)
    return maps.reshape(h, w, 4), valid.reshape(h, w)


def foreground_cells(boxes, scores, feature, bounds, settings):
    """Highest-score ownership, deterministic ties; preserve full instance geometry."""
    h, w = feature.shape[-2:]
    centers = cell_centers(h, w, bounds, feature).reshape(-1, 3)
    owner = torch.full((h*w,), -1, dtype=torch.long, device=feature.device)
    foreground = torch.zeros(h*w, dtype=torch.bool, device=feature.device)
    ambiguous = torch.zeros_like(foreground)
    used = 0
    # Boxes are already in stable descending score order.
    for i, box in enumerate(boxes):
        delta = centers[:, :2]-box[:2]
        c, s = box[6].cos(), box[6].sin()
        x, y = c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]
        inside = (x.abs() <= box[5]/2+settings['foreground_margin_m']) & (y.abs() <= box[4]/2+settings['foreground_margin_m'])
        ambiguous |= inside & foreground
        foreground |= inside
        ids = torch.where(inside & (owner < 0))[0]
        budget = min(settings['cells_per_instance'], settings['max_cells']-used)
        if budget <= 0:
            continue
        if len(ids) > budget:
            # Uniform grid-order thinning retains both sides of each box.
            ids = ids[torch.linspace(0, len(ids)-1, budget, device=ids.device).long()]
        owner[ids] = i
        used += len(ids)
    ids = torch.where(owner >= 0)[0]
    indices = torch.stack((ids//w, ids % w), -1).to(torch.int32)
    points = centers[ids].clone()
    points[:, 2] = boxes[owner[ids], 2]
    features = feature.permute(1, 2, 0).reshape(-1, feature.shape[0])[ids]
    return dict(indices=indices, instance_ids=owner[ids].to(torch.int32), points=points,
                features=features, foreground=foreground.reshape(h, w), ambiguity=int(ambiguous.sum()))


class SfrSender(PointPillarCodyntrustSingle):
    def __init__(self, args, settings):
        super().__init__(args)
        self.settings = settings
        self.bounds = args['lidar_range']
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Fixed sender BN statistics even when an owning module enters train().
        return super().train(False)

    @torch.no_grad()
    def extract(self, processed, frame_count, anchors, postprocessor):
        raw, coords = processed['voxel_features'], processed['voxel_coords']
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
        output = {'psm_single': self.cls_head(features), 'rm_single': self.reg_head(features)}
        if self.use_dir:
            output['dm_single'] = self.dir_head(features)
        predictions = postprocessor.single_post_process(output,
            torch.eye(4, device=raw.device).repeat(frame_count, 1, 1),
            torch.zeros(frame_count, device=raw.device), anchors, num_sweeps=frame_count, num_roi_thres=-1)
        result = []
        for i in range(frame_count):
            pred = predictions[i]
            # Stable tie order on the deployed legacy torch version.
            order = torch.as_tensor(np.argsort(-pred['scores'].detach().cpu().numpy(), kind='stable'), device=raw.device)[:self.settings['max_instances']]
            boxes, scores = pred['pred_box_center_tensor'][order], pred['scores'][order]
            sparse = foreground_cells(boxes, scores, features[i], self.bounds, self.settings)
            keep = coords[:, 0] == i
            maps, valid = geometry_map(raw[keep], processed['voxel_num_points'][keep], features.shape[-2:], self.bounds)
            result.append(dict(boxes=boxes, scores=scores, geometry=maps, observed=valid, **sparse))
        return features, result

    def wire_message(self, message, metadata):
        """All receiver data passes through the actual codec, including evaluation."""
        qmap, map_scale = quantize(message['geometry'])
        qfeatures, feature_scale = quantize(message['features'])
        tensors = {key: message[key] for key in ('boxes', 'scores', 'indices', 'instance_ids', 'observed', 'foreground')}
        tensors.update(qmap=qmap, map_scale=map_scale, qfeatures=qfeatures, feature_scale=feature_scale)
        metadata = dict(metadata, codec_version='sfr-1', bounds=list(self.bounds),
                        feature_shape=list(message['foreground'].shape), axes='y,x; metric xyz; hwl boxes',
                        map_channels=['occupied', 'log_retained_density', 'mean_z', 'std_z'])
        packet, accounting = encode(metadata, tensors)
        metadata, received = decode(packet, message['features'].device)
        received['geometry'] = received.pop('qmap').float()*received.pop('map_scale')
        received['features'] = received.pop('qfeatures').float()*received.pop('feature_scale')
        received['instance_ids'] = received['instance_ids'].long()
        h, w = metadata['feature_shape']
        grid = cell_centers(h, w, self.bounds, received['features'])
        rows, cols = received['indices'].long().unbind(-1)
        received['points'] = grid[rows, cols].clone()
        received['points'][:, 2] = received['boxes'][received['instance_ids'], 2]
        received['metadata'] = metadata
        return received, accounting
