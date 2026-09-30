"""INT FM-only adapted to OpenCOOD's P1 Pillars encoder and detection head."""
import torch
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle
from opencood.models.sub_modules.int_feature_memory import INTFeatureMemory
from opencood.models.sub_modules.int_spike_memory import INTSpikeMemory


class PointPillarINT(PointPillarCodyntrustSingle):
    def __init__(self, args, mode='concat'):
        super().__init__(args)
        if mode in ('lif', 'leaky'):
            self.feature_memory = INTSpikeMemory(mode, 64, args['lidar_range'],
                                                 **args.get('int_memory', {}))
        else:
            self.feature_memory = INTFeatureMemory(mode, 64, args['lidar_range'])

    def load_single_checkpoint(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('feature_memory.')}
        if expected != set(state):
            raise ValueError('Single-frame checkpoint keys do not exactly match the spatial detector')
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('feature_memory.') for k in missing):
            raise ValueError((missing, unexpected))
        return len(expected)

    def step(self, data, state, meta, reset=False, align=True, detach_state=True):
        p = data['processed_lidar']
        if p['voxel_features'].shape[0] == 0:
            nx, ny, _ = self.scatter.model_cfg['grid_size']
            spatial = self.cls_head.weight.new_zeros((1, 64, int(ny), int(nx)))
        else:
            batch = {k: p[k] for k in ['voxel_features', 'voxel_coords', 'voxel_num_points']}
            batch['batch_size'] = 1
            spatial = self.scatter(self.pillar_vfe(batch))['spatial_features']
        temporal, next_state, stats = self.feature_memory.step(
            spatial, state, meta, reset, align, detach_state=detach_state)
        batch = self.backbone({'spatial_features': temporal})
        features = batch['spatial_features_2d']
        if self.shrink_flag:
            features = self.shrink_conv(features)
        prediction = {'psm': self.cls_head(features), 'rm': self.reg_head(features)}
        if self.use_dir:
            prediction['dm'] = self.dir_head(features)
        return prediction, next_state, stats

    def forward(self, data):
        raise RuntimeError('Use step(data, scene_state, meta) via run_int_ego.py; forward would discard history.')
