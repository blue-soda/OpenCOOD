"""Opt-in CUDA optimization for CoDynTrust MC inference.

The original CoDynTrust implementation executes the extra MC-dropout samples
one after another in Python.  This module keeps the original model untouched
and batches those samples along the leading CAV dimension.  Dropout remains
active for every batch item, so the samples are still independent MC samples;
only their execution schedule changes.
"""

import torch
import torch.nn as nn

from opencood.models.point_pillar_codyntrust import PointPillarCodyntrust


class _MCBatchContext:
    def __init__(self, sample_count, chunk_size, final_key):
        self.sample_count = sample_count
        self.chunk_size = max(1, min(chunk_size, sample_count))
        self.final_key = final_key
        self.initial_backbone_done = False
        self.mc_started = False
        self.replay_active = False
        self.current_index = -1
        self.sample_batch = None
        self.features = None
        self.outputs = {}

    def split(self, tensor):
        return list(tensor.split(self.sample_batch, dim=0))


class _BackboneReplay(nn.Module):
    """Run all stochastic backbone samples in one or a few CUDA batches."""

    def __init__(self, inner, context):
        super().__init__()
        self.inner = inner
        self.context = context

    def forward(self, data_dict):
        ctx = self.context
        if not ctx.initial_backbone_done:
            ctx.initial_backbone_done = True
            return self.inner(data_dict)

        if not ctx.mc_started:
            ctx.mc_started = True
            ctx.replay_active = True
        elif not ctx.replay_active:
            return self.inner(data_dict)

        if ctx.features is None:
            base_features = data_dict['spatial_features']
            ctx.sample_batch = base_features.shape[0]
            feature_batches = []
            for start in range(0, ctx.sample_count, ctx.chunk_size):
                count = min(ctx.chunk_size, ctx.sample_count - start)
                expanded = dict(data_dict)
                expanded['spatial_features'] = base_features.repeat(count, 1, 1, 1)
                expanded = self.inner(expanded)
                feature_batches.extend(ctx.split(expanded['spatial_features_2d']))
            ctx.features = feature_batches

        ctx.current_index += 1
        if ctx.current_index >= ctx.sample_count:
            raise RuntimeError('CoDynTrust MC replay received too many samples')
        output = dict(data_dict)
        output['spatial_features_2d'] = ctx.features[ctx.current_index]
        return output


class _HeadReplay(nn.Module):
    """Batch one repeated MC head (shrink, cls, reg, dir, or uncertainty)."""

    def __init__(self, inner, context, name, source_name):
        super().__init__()
        self.inner = inner
        self.context = context
        self.name = name
        self.source_name = source_name

    def forward(self, tensor):
        ctx = self.context
        if not ctx.replay_active:
            return self.inner(tensor)

        if self.name not in ctx.outputs:
            source = ctx.features if self.source_name == 'features' else ctx.outputs[self.source_name]
            batched = self.inner(torch.cat(source, dim=0))
            ctx.outputs[self.name] = ctx.split(batched)

        result = ctx.outputs[self.name][ctx.current_index]
        if self.name == ctx.final_key and ctx.current_index == ctx.sample_count - 1:
            # The next operation after the MC loop is the normal fusion path.
            # Let the wrapped modules behave normally there.
            ctx.replay_active = False
        return result


class PointPillarCodyntrustOptimized(PointPillarCodyntrust):
    """CoDynTrust with opt-in batched MC-dropout inference."""

    def __init__(self, args):
        super().__init__(args)
        self.mc_batch_replay = args.get('mc_batch_replay', True)
        self.mc_batch_chunk = args.get('mc_batch_chunk', self.inference_num - 1)

    def forward(self, data_dict, dataset=None):
        if not self.mc_batch_replay or not self.inference_state or self.inference_num <= 1:
            return super().forward(data_dict, dataset)

        final_key = 'unc_cls' if self.re_parameterization else 'unc'
        context = _MCBatchContext(
            sample_count=self.inference_num - 1,
            chunk_size=self.mc_batch_chunk,
            final_key=final_key,
        )

        originals = {
            'backbone': self.backbone,
            'shrink_conv': self.shrink_conv if self.shrink_flag else None,
            'cls_head': self.cls_head,
            'reg_head': self.reg_head,
            'unc_head': self.unc_head,
            'dir_head': self.dir_head if self.use_dir else None,
            'unc_head_cls': self.unc_head_cls if self.re_parameterization else None,
        }

        self.backbone = _BackboneReplay(originals['backbone'], context)
        if self.shrink_flag:
            self.shrink_conv = _HeadReplay(originals['shrink_conv'], context, 'shrink', 'features')
        self.cls_head = _HeadReplay(originals['cls_head'], context, 'cls', 'shrink' if self.shrink_flag else 'features')
        self.reg_head = _HeadReplay(originals['reg_head'], context, 'reg', 'shrink' if self.shrink_flag else 'features')
        self.unc_head = _HeadReplay(originals['unc_head'], context, 'unc', 'shrink' if self.shrink_flag else 'features')
        if self.use_dir:
            self.dir_head = _HeadReplay(originals['dir_head'], context, 'dir', 'shrink' if self.shrink_flag else 'features')
        if self.re_parameterization:
            self.unc_head_cls = _HeadReplay(originals['unc_head_cls'], context, 'unc_cls', 'shrink' if self.shrink_flag else 'features')

        # The first backbone call and first head pass are the original
        # single-sample path.  Replay starts at the second backbone call.
        context.replay_active = False
        try:
            return super().forward(data_dict, dataset)
        finally:
            # Restore the exact module graph, including for exceptions.
            self.backbone = originals['backbone']
            if self.shrink_flag:
                self.shrink_conv = originals['shrink_conv']
            self.cls_head = originals['cls_head']
            self.reg_head = originals['reg_head']
            self.unc_head = originals['unc_head']
            if self.use_dir:
                self.dir_head = originals['dir_head']
            if self.re_parameterization:
                self.unc_head_cls = originals['unc_head_cls']
