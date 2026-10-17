"""Inference reference: ego-warped stage caches with actual cropped execution.

Stage refreshes are exact relative to their (possibly approximate) input. No
teacher output is injected into skipped cells. All costs include crop halos.
"""
import math
import torch
import torch.nn.functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev


def tile_mask(mask, tile):
    return F.max_pool2d(mask.float(), tile, tile, ceil_mode=True) > 0


def expand_tiles(mask, tile, shape):
    return mask.repeat_interleave(tile, -2).repeat_interleave(tile, -1)[..., :shape[0], :shape[1]]


def select_updates(current, previous, valid, fraction, tile=32, occupancy_policy='cell'):
    """Cheap heuristic only: relative difference plus occupancy/coverage safety."""
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0,1]')
    pool = lambda x: F.avg_pool2d(x, tile, tile, ceil_mode=True, count_include_pad=False)
    error = pool((current-previous).square().mean(1, keepdim=True))
    energy = torch.maximum(pool(current.square().mean(1, keepdim=True)),
                           pool(previous.square().mean(1, keepdim=True)))
    score = (error / energy.clamp_min(1e-6)).sqrt()
    selected = torch.zeros_like(score, dtype=torch.bool)
    count = int(math.ceil(fraction * score.numel()))
    if count:
        indices = score.flatten().topk(count, sorted=False).indices
        selected.view(-1)[indices] = score.flatten()[indices] > 0
    changed_occupancy = (current.abs().sum(1, keepdim=True)>0) != (previous.abs().sum(1, keepdim=True)>0)
    if occupancy_policy == 'tile':
        changed_occupancy = expand_tiles(
            tile_mask(current.abs().sum(1,keepdim=True)>0,tile) !=
            tile_mask(previous.abs().sum(1,keepdim=True)>0,tile),tile,current.shape[-2:])
    elif occupancy_policy != 'cell':
        raise ValueError('Unknown occupancy policy')
    return expand_tiles(selected, tile, current.shape[-2:]) | changed_occupancy | ~valid


def rectangles(active):
    """Merge horizontal tile runs and identical runs on consecutive rows."""
    rows=active[0,0].cpu().tolist()
    result=[];previous={}
    for y,row in enumerate(rows):
        current={};x=0
        while x<len(row):
            if not row[x]:
                x+=1;continue
            start=x
            while x<len(row) and row[x]:x+=1
            key=(start,x)
            if key in previous:
                index=previous[key];result[index][2]=y+1
            else:
                index=len(result);result.append([y,start,y+1,x])
            current[key]=index
        previous=current
    return result


def stage_dependency_work(module, output_mask, input_shape):
    """Ideal deduplicated convolution work for the frozen BasicBlock stack.

Counts MACs only; this is NOT an implemented sparse kernel or latency estimate.
Backward dependency propagation includes both main and residual branches.
"""
    required = output_mask
    needed_macs = dense_macs = 0.
    blocks = list(module)
    for index in reversed(range(len(blocks))):
        block = blocks[index]
        conv1, conv2 = block.conv1, block.conv2
        assert conv1.kernel_size == conv2.kernel_size == (3,3)
        assert conv2.stride == (1,1) and conv1.dilation == conv2.dilation == (1,1)
        assert conv1.groups == conv2.groups == 1
        n = float(required.sum());total = required.numel()
        per_pixel = 9*conv2.in_channels*conv2.out_channels
        needed_macs += n*per_pixel;dense_macs += total*per_pixel
        # conv2 needs all conv1 outputs in its 3x3 neighborhood.
        intermediate = F.max_pool2d(required.float(),3,1,1)>0
        per_pixel = 9*conv1.in_channels*conv1.out_channels
        needed_macs += float(intermediate.sum())*per_pixel;dense_macs += total*per_pixel
        if block.downsample is not None:
            down = block.downsample[0]
            assert down.kernel_size == (1,1)
            per_pixel = down.in_channels*down.out_channels
            needed_macs += n*per_pixel;dense_macs += total*per_pixel
        stride = conv1.stride[0]
        shape = input_shape if index==0 else output_mask.shape[-2:]
        base = [(intermediate.shape[-2+j]-1)*stride+1 for j in range(2)]
        padding = tuple(int(shape[j]-base[j]) for j in range(2))
        required = F.conv_transpose2d(intermediate.float(),intermediate.new_ones((1,1,3,3),dtype=torch.float),
                                      stride=stride,padding=1,output_padding=padding)>0
        # The residual input locations are a subset of the expanded main path.
    stats = dict(ideal_required_conv_macs=needed_macs,dense_conv_macs=dense_macs,
                 ideal_conv_mac_fraction=needed_macs/dense_macs,
                 unique_input_support_fraction=float(required.float().mean()))
    return stats, required


def refresh_tiles(module, x, cached, requested, stride, radius, tile=8,
                  fallback_fraction=.8, allow_fallback=True):
    """Run only selected stage-output tiles, batching equal-shaped input crops.

Only eval-mode local convolutional modules are supported. radius is the full
input-space dependency radius about output index * stride, including residuals.
Input crop origins are stride aligned. Do not externally zero-pad global edges:
intermediate BN offsets would then produce a different boundary condition.
"""
    if module.training or x.shape[0] != 1:
        raise ValueError('Frozen eval modules and batch=1 are required')
    h, w = x.shape[-2:]
    oh, ow = cached.shape[-2:]
    assert (oh, ow) == ((h+stride-1)//stride, (w+stride-1)//stride)
    assert requested.shape == cached[:, :1].shape
    active = tile_mask(requested, tile)
    written = expand_tiles(active, tile, (oh, ow))
    regions = rectangles(active)
    jobs = []
    area = 0
    for ty, tx, ey, ex in regions:
        y0, x0 = ty*tile, tx*tile
        y1, x1 = min(ey*tile, oh), min(ex*tile, ow)
        iy0 = max(0, ((y0*stride-radius)//stride)*stride)
        ix0 = max(0, ((x0*stride-radius)//stride)*stride)
        iy1 = min(h, math.ceil(((y1-1)*stride+radius+1)/stride)*stride)
        ix1 = min(w, math.ceil(((x1-1)*stride+radius+1)/stride)*stride)
        jobs.append((y0,x0,y1,x1,iy0,ix0,iy1,ix1))
        area += (iy1-iy0)*(ix1-ix0)
    ratio = area / float(h*w)
    stats = dict(requested_fraction=float(requested.float().mean()),
                 selected_fraction=float(written.float().mean()),
                 crop_area_ratio_before_fallback=ratio, tiles=len(jobs), fallback=False,
                 executed_input_area_ratio=ratio, actual_write_fraction=float(written.float().mean()))
    if allow_fallback and (bool(active.all()) or ratio >= fallback_fraction):
        stats.update(fallback=True, executed_input_area_ratio=1., actual_write_fraction=1.)
        return module(x), torch.ones_like(requested), stats
    output = cached.clone()
    groups = {}
    for job in jobs:
        groups.setdefault((job[6]-job[4], job[7]-job[5]), []).append(job)
    for group in groups.values():
        for start in range(0, len(group), 8):
            batch = group[start:start+8]
            crops = torch.cat([x[..., j[4]:j[6], j[5]:j[7]] for j in batch], 0)
            result = module(crops)
            for i, (y0,x0,y1,x1,iy0,ix0,_,_) in enumerate(batch):
                sy, sx = y0-iy0//stride, x0-ix0//stride
                output[..., y0:y1, x0:x1] = result[i:i+1, :, sy:sy+y1-y0, sx:sx+x1-x0]
    return output, written, stats


class CacheRepairDetector:
    """Fixed existing detector; full FM, cached ResNet stages, dense decoder/head."""
    def __init__(self, model, config, fraction=.5, tile=8, max_age=1., dense=False, occupancy_policy='cell', oracle_stage0=False, support_profile=False):
        self.model, self.config = model, config
        self.fraction, self.tile, self.max_age, self.dense = fraction, tile, max_age, dense
        self.occupancy_policy = occupancy_policy
        self.oracle_stage0 = oracle_stage0
        self.support_profile = support_profile
        args = config['model']['args']
        self.lidar_range = args['lidar_range']
        self.layers = args['base_bev_backbone']['layer_nums']
        assert args['base_bev_backbone']['layer_strides'] == [2,2,2]
        assert args['base_bev_backbone']['resnet']
        assert not model.backbone.use_dropout

    def step(self, data, state, meta):
        m = self.model
        p = data['processed_lidar']
        if p['voxel_features'].shape[0]:
            batch = {k:p[k] for k in ['voxel_features','voxel_coords','voxel_num_points']}
            batch['batch_size'] = 1
            spatial = m.scatter(m.pillar_vfe(batch))['spatial_features']
        else:
            nx, ny, _ = m.scatter.model_cfg['grid_size']
            spatial = m.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
        temporal, fm, info = m.feature_memory.step(spatial, None if state is None else state['fm'], meta)
        fresh = state is None or bool(info['reset'])
        if fresh or self.dense:
            previous = torch.zeros_like(spatial)
            requests = torch.ones_like(spatial[:,:1], dtype=torch.bool)
        else:
            previous, valid = warp_bev(state['shallow'], state['pose'], meta['pose'], self.lidar_range)
            requests = select_updates(spatial, previous, valid, self.fraction, occupancy_policy=self.occupancy_policy)
        stages, ages, stats = [], [], []
        x = temporal
        first_written = None
        for i, n in enumerate(self.layers):
            module = getattr(m.backbone.resnet, 'layer%d' % i)
            input_shape = x.shape[-2:]
            if fresh or self.dense or (self.oracle_stage0 and i>0):
                x = module(x)
                age = torch.zeros_like(x[:,:1])
                written = torch.ones_like(age, dtype=torch.bool)
                st = dict(fallback=True, executed_input_area_ratio=1., actual_write_fraction=1.,
                          requested_fraction=1., selected_fraction=1., crop_area_ratio_before_fallback=1., tiles=0)
            else:
                cached, valid = warp_bev(state['stages'][i], state['pose'], meta['pose'], self.lidar_range)
                age, _ = warp_bev(state['ages'][i], state['pose'], meta['pose'], self.lidar_range)
                age = age + info['delta_t']
                req = F.max_pool2d(requests.float(), 2**(i+1), 2**(i+1), ceil_mode=True)>0
                if self.oracle_stage0 and i==0:
                    # Diagnostic upper-information selector, NOT deployable inference.
                    # Dense output is used only for ranking; skipped outputs still
                    # come from recursively warped cache, never this dense tensor.
                    oracle = module(x)
                    error = F.avg_pool2d((oracle-cached).square().mean(1,keepdim=True),
                                         self.tile,self.tile,ceil_mode=True,count_include_pad=False)
                    chosen = torch.zeros_like(error,dtype=torch.bool)
                    count = int(math.ceil(self.fraction*error.numel()))
                    if count:chosen.view(-1)[error.flatten().topk(count,sorted=False).indices]=True
                    req = expand_tiles(chosen,self.tile,cached.shape[-2:])
                req = req | ~valid | (age >= self.max_age)
                # BasicBlock stack: first block radius=3, each further block adds 4.
                x, written, st = refresh_tiles(module, x, cached, req, 2, 4*n-1, self.tile,
                                               allow_fallback=not self.oracle_stage0)
                if self.oracle_stage0:
                    st['executed_input_area_ratio'] += 1.  # Include dense oracle work.
                age = torch.where(written, torch.zeros_like(age), age)
            if self.support_profile:
                work, _ = stage_dependency_work(module,written,input_shape)
                st.update(work)
            if i == 0:
                first_written = written.repeat_interleave(2,-2).repeat_interleave(2,-1)[...,:spatial.shape[-2],:spatial.shape[-1]]
            stages.append(x); ages.append(age); stats.append(st)
        # Decoder uses actual mixed caches, never dense teacher outputs.
        features = m.backbone.decode_multiscale_feature(stages)
        if m.shrink_flag:
            features = m.shrink_conv(features)
        pred = dict(psm=m.cls_head(features), rm=m.reg_head(features))
        if m.use_dir:
            pred['dm'] = m.dir_head(features)
        new = dict(fm=fm, pose=meta['pose'], stages=stages, ages=ages,
                   shallow=torch.where(first_written, spatial, previous))
        info.update(cache_stages=stats, cache_bytes=sum(t.numel()*t.element_size() for t in
                    stages+ages+[new['shallow']])+fm['value'].numel()*fm['value'].element_size())
        return pred, new, info
