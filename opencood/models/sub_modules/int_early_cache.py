"""Early same-layer change gates and whole-frame downstream feature reuse.

Only one intervening frame may be reused. Thus an eligible GRU comparison's
previous hidden state and the deep cache always refer to the same source frame.
No weights or original detection operations are changed.
"""
import math
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_gate import descriptor_change

VERSION = 'early_hidden_or_stage1_period2_strict_v1'


def early_descriptor(x, pool):
    b, c, h, w = x.shape
    assert c % 16 == 0
    return F.avg_pool2d(x.reshape(b, 16, c // 16, h, w).mean(2), pool, pool,
                        ceil_mode=True, count_include_pad=False)


def refresh_reason(reset, count, age, valid, score, threshold, period=2):
    if reset:
        return 'initial'
    if age <= 0:
        raise ValueError('Non-causal early cache')
    if count >= period:
        return 'period'
    if age >= .25:
        return 'age'
    if score > threshold:
        return 'change'
    if not valid:
        return 'coverage'
    return None


class EarlyCacheDetector:
    def __init__(self, model, config, gate='gru', cache_kind='stages', threshold=0.,
                 period=2, probe=False):
        if gate not in ('gru', 'stage1') or cache_kind not in ('stages', 'decoded'):
            raise ValueError('Unknown early-cache combination')
        if period not in (1, 2) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError('Finite threshold and period 1/2 required')
        if model.feature_memory.mode != 'gru' or model.training:
            raise ValueError('Frozen ANN GRU required')
        self.model, self.bounds = model, config['model']['args']['lidar_range']
        self.gate, self.cache_kind = gate, cache_kind
        self.threshold, self.period, self.probe = threshold, period, probe

    def step(self, data, state, meta):
        m = self.model
        p = data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch = {k: p[k] for k in ('voxel_features', 'voxel_coords', 'voxel_num_points')}
            batch['batch_size'] = 1
            spatial = m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx, ny, _ = m.scatter.model_cfg['grid_size']
            spatial = m.cls_head.weight.new_zeros(1, 64, int(ny), int(nx))
        cache = None if state is None else state['cache']
        count = 0 if cache is None else cache['frames'] + 1
        age = 0. if cache is None else (int(meta['timestamp_us']) - cache['timestamp_us']) / 1e6
        check = cache is not None and (self.probe or (count < self.period and age < .25))
        observed = {}

        def observe(current, past, valid):
            observed['valid'] = bool(valid.all())
            observed['score'] = descriptor_change(early_descriptor(current, 8), early_descriptor(past, 8))

        x, fm, info = m.feature_memory.step(spatial, None if state is None else state['fm'], meta,
            state_observer=observe if check and self.gate == 'gru' else None)
        fresh = cache is None or bool(info['reset'])
        if fresh:
            cache = None
            check = False
            count = 0
        stages = []
        executed = [0., 0., 0.]
        descriptor = None
        if self.gate == 'stage1':
            x = m.backbone.resnet.layer0(x)
            stages.append(x)
            executed[0] = 1.
            # Save a reference only at refresh. One common physical descriptor grid.
            descriptor = early_descriptor(x, 4)
            if check:
                aligned, valid = warp_bev(cache['descriptor'], cache['pose'], meta['pose'], self.bounds)
                observed = dict(valid=bool(valid.all()), score=descriptor_change(descriptor, aligned))
        score = observed.get('score')
        valid = observed.get('valid', False)
        warped = []
        # In probe mode, inspect the previous full frame but always execute densely.
        # In inference, never compute deep current features to decide whether to skip.
        if check and valid and (self.probe or score <= self.threshold):
            for i, value in enumerate(cache['values']):
                if self.cache_kind == 'stages' and self.gate == 'stage1' and i == 0:
                    warped.append(stages[0])
                    continue
                aligned, covered = warp_bev(value, cache['pose'], meta['pose'], self.bounds)
                valid = valid and bool(covered.all())
                warped.append(aligned)
        reason = refresh_reason(fresh, count, age, valid, score, self.threshold, self.period)
        if self.probe:
            reason = 'probe'
        if reason is not None:
            for i in range(len(stages), 3):
                x = getattr(m.backbone.resnet, 'layer%d' % i)(x)
                stages.append(x)
                executed[i] = 1.
            features = m.backbone.decode_multiscale_feature(stages)
            if m.shrink_flag:
                features = m.shrink_conv(features)
            values = stages if self.cache_kind == 'stages' else [features]
            cache = dict(values=values, pose=meta['pose'], timestamp_us=int(meta['timestamp_us']), frames=0)
            if descriptor is not None:
                cache['descriptor'] = descriptor
            decoder = 1
        else:
            # No storage of rewarped caches: retain the last real refresh anchor.
            cache = dict(cache, frames=count)
            if self.cache_kind == 'stages':
                features = m.backbone.decode_multiscale_feature(warped)
                if m.shrink_flag:
                    features = m.shrink_conv(features)
                decoder = 1
            else:
                features = warped[0]
                decoder = 0
        pred = dict(psm=m.cls_head(features), rm=m.reg_head(features))
        if m.use_dir:
            pred['dm'] = m.dir_head(features)
        tensors = cache['values'] + [fm['value']]
        if 'descriptor' in cache:
            tensors.append(cache['descriptor'])
        info.update(cache_stages=[dict(actual_write_fraction=v, executed_input_area_ratio=v) for v in executed],
            cache_bytes=sum(t.numel()*t.element_size() for t in tensors),
            early_refresh_reason=reason, early_score=score, candidate_valid=valid,
            candidate_age=age, decoder_called=decoder, extra_gru_alignment_calls=0)
        return pred, dict(fm=fm, cache=cache), info
