"""Immutable full-refresh anchors with local residual repair and spatial readout."""
import torch
from torch import nn
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.models.sub_modules.int_feature_memory import warp_bev


class AnchorResidualModules(nn.Module):
    def __init__(self, fusion=False):
        super().__init__()
        self.repair = LightRepair(256, width=64)
        # Both arms have identically initialized repair weights. The readout
        # starts as identity and has only radius-one spatial dependencies.
        self.readout = LightRepair(256, width=96) if fusion else None

    def decode(self, evidence, value, valid):
        if self.readout is None:
            return value
        return self.readout(torch.cat([evidence, value, valid.to(value.dtype)], 1))


def refresh_reason(cache, meta, reset, period):
    if cache is None or reset:
        return 'reset'
    age = (int(meta['timestamp_us']) - cache['stamp']) / 1e6
    if age <= 0:
        raise ValueError('Noncausal anchor stream')
    if age >= .25:
        return 'age'
    if cache['frames'] + 1 >= period:
        return 'period'
    return None


def predict(model, features):
    result = dict(psm=model.cls_head(features), rm=model.reg_head(features))
    if model.use_dir:
        result['dm'] = model.dir_head(features)
    return result


class AnchorResidualDetector:
    """One cached H; full encoder/decoder runs only on refresh frames.

    FM and stage1 run every frame. Repaired H and readout results never overwrite
    the anchor. This first experiment executes the small modules over the whole
    grid; spatially selective execution is a separate subsequent intervention.
    """
    def __init__(self, model, config, modules, period=3):
        if model.training or period not in (1, 2, 3):
            raise ValueError('Frozen encoder and period 1/2/3 required')
        self.model, self.modules, self.period = model, modules, period
        self.bounds = config['model']['args']['lidar_range']

    def step(self, data, state, meta):
        m = self.model
        p = data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch = {k:p[k] for k in ('voxel_features', 'voxel_coords', 'voxel_num_points')}
            batch['batch_size'] = 1
            spatial = m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx, ny, _ = m.scatter.model_cfg['grid_size']
            spatial = m.cls_head.weight.new_zeros(1, 64, int(ny), int(nx))
        temporal, fm, info = m.feature_memory.step(spatial, None if state is None else state['fm'], meta)
        evidence = m.backbone.resnet.layer0(temporal)
        cache = None if state is None else state['cache']
        reason = refresh_reason(cache, meta, info['reset'], self.period)
        if reason:
            second = m.backbone.resnet.layer1(evidence)
            third = m.backbone.resnet.layer2(second)
            value = m.backbone.decode_multiscale_feature([evidence, second, third])
            if m.shrink_flag:
                value = m.shrink_conv(value)
            cache = dict(value=value, pose=meta['pose'], stamp=int(meta['timestamp_us']), frames=0)
            valid = torch.ones_like(value[:, :1], dtype=torch.bool)
        else:
            old, valid = warp_bev(cache['value'], cache['pose'], meta['pose'], self.bounds)
            value = self.modules.repair(torch.cat([evidence, old, valid.to(old.dtype)], 1))
            cache = dict(cache, frames=cache['frames'] + 1)
        if evidence.shape[-2:] != value.shape[-2:]:
            raise ValueError('Evidence and H must share the same lattice')
        features = self.modules.decode(evidence, value, valid)
        prediction = predict(m, features)
        info.update(anchor_partial=reason is None, refresh_reason=reason,
                    anchor_stamp=cache['stamp'], anchor_depth=cache['frames'],
                    anchor_age=(int(meta['timestamp_us'])-cache['stamp'])/1e6,
                    anchor_bytes=cache['value'].numel()*cache['value'].element_size(),
                    repair_area_fraction=0. if reason else 1.,
                    original_deep_call_fraction=1. if reason else 0.)
        return prediction, dict(fm=fm, cache=cache), info
