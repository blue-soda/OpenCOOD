"""Frozen quantized fine residual encoder with real static/delta execution."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_integer_event_reference import folded_integer_weights, integer_input, dense_integer_reference
from opencood.models.sub_modules.int_strided_integer_events import strided_integer_events
from opencood.models.sub_modules.int_compact_causal_integer_state import narrow_integer_state
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector, reset_reason


FINE_POLICY = {'0.conv1': 'static', '0.conv2': 'delta', '0.downsample.0': 'static',
               '1.conv1': 'delta', '1.conv2': 'delta', '2.conv1': 'delta', '2.conv2': 'static'}


class FineIntegerConv(nn.Module):
    def __init__(self, conv, quantum, mode):
        super().__init__()
        if mode not in ('dense', 'static', 'delta'): raise ValueError(mode)
        self.source = copy.deepcopy(conv)
        q = torch.as_tensor(quantum, dtype=torch.float64, device=conv.weight.device).reshape(1, -1, 1, 1)
        if q.numel() != conv.in_channels or not bool(torch.isfinite(q).all()) or not bool((q > 0).all()): raise ValueError('Invalid calibration')
        weight, scale = folded_integer_weights(conv, q)
        self.register_buffer('quantum', q); self.register_buffer('integer_weight', weight); self.register_buffer('scale', scale)
        self.mode = mode; self.previous = None; self.next_state = None
        self.capture_audit = False; self.audit = None; self.statistics = {}; self.fanout = None

    def forward(self, x):
        if self.training or torch.is_grad_enabled() or x.shape[0] != 1: raise ValueError('Frozen single-stream inference only')
        conv = self.source; n = integer_input(x, self.quantum)
        rebuild = self.mode == 'delta' and self.previous is None
        if self.mode == 'delta' and not rebuild:
            old, prior = self.previous
            if n.shape != old.shape: raise ValueError('Grid changed without reset')
            delta = n - old.to(torch.int64)
        else: delta = n
        if self.mode == 'dense':
            acc, bound = dense_integer_reference(conv, n, self.integer_weight); execution = None
        else:
            update, execution = strided_integer_events(conv, delta, self.integer_weight)
            acc = update if self.mode == 'static' or rebuild else prior.to(torch.int64) + update
            bound = execution['absolute_accumulation_bound']
        if bool((acc.abs() >= 2**54).any()): raise ValueError('Accumulator state range exceeded')
        if self.mode == 'delta':
            self.next_state = (narrow_integer_state(n, True), narrow_integer_state(acc))
        else: self.next_state = None
        h, w = acc.shape[-2:]; kh, kw = conv.kernel_size; sh, sw = conv.stride; ph, pw = conv.padding
        if self.fanout is None:
            op = (x.shape[-2] - ((h-1)*sh-2*ph+kh), x.shape[-1] - ((w-1)*sw-2*pw+kw))
            self.fanout = F.conv_transpose2d(x.new_ones(1, 1, h, w), x.new_ones(1, 1, kh, kw), stride=conv.stride, padding=conv.padding, output_padding=op).to(torch.int64)
        if self.fanout.shape[-2:] != x.shape[-2:]: raise ValueError('Static grid required')
        mac = int(((delta != 0).to(torch.int64) * self.fanout).sum()) * conv.out_channels
        ac = int((delta.abs() * self.fanout).sum()) * conv.out_channels
        dense = acc.numel() * conv.in_channels * kh * kw
        if execution is not None and mac != execution['executed_macs']: raise RuntimeError('Event count mismatch')
        self.statistics = dict(mode=self.mode, rebuild=rebuild, input_hw=list(x.shape[-2:]), output_hw=[h,w],
            input_channels=conv.in_channels, output_channels=conv.out_channels, kernel=list(conv.kernel_size), stride=list(conv.stride),
            dense_mac=dense, actual_fp64_flops=2*dense if self.mode=='dense' else 0, actual_integer_macs=0 if self.mode=='dense' else mac,
            event_mac=mac, unit_spike_ac=ac, nonzero_events=int((delta != 0).sum()), unit_spikes=int(delta.abs().sum()),
            max_input_integer=int(n.abs().max()), max_delta_integer=int(delta.abs().max()), max_accumulator=int(acc.abs().max()), accumulation_bound=bound,
            quantization_error_in_quanta=float((x.double()/self.quantum-n.double()).abs().max()),
            reference_dtype=None if self.next_state is None else str(self.next_state[0].dtype), accumulator_dtype=None if self.next_state is None else str(self.next_state[1].dtype),
            reference_bytes=0 if self.next_state is None else self.next_state[0].numel()*self.next_state[0].element_size(),
            accumulator_bytes=0 if self.next_state is None else self.next_state[1].numel()*self.next_state[1].element_size(),
            derived_static_bytes=self.integer_weight.numel()*self.integer_weight.element_size()+self.scale.numel()*self.scale.element_size()+self.quantum.numel()*self.quantum.element_size()+self.fanout.numel()*self.fanout.element_size())
        self.audit = (n, acc) if self.capture_audit else None
        y = (acc.double() * self.scale.reshape(1,-1,1,1)).to(x.dtype)
        return y if conv.bias is None else y + conv.bias.reshape(1,-1,1,1)


class FineIntegerEncoder(nn.Module):
    def __init__(self, source, calibration, mode):
        super().__init__(); self.layers = copy.deepcopy(source)
        if mode not in ('dense', 'static', 'hybrid'): raise ValueError(mode)
        convs = {n:m for n,m in self.layers.named_modules() if isinstance(m,nn.Conv2d)}
        if set(convs) != set(FINE_POLICY) or set(calibration) != set(convs): raise ValueError('Three fine residual blocks required')
        for name, conv in convs.items():
            parent, _, leaf = name.rpartition('.')
            self.layers.get_submodule(parent)._modules[leaf] = FineIntegerConv(conv, calibration[name]['quanta'], FINE_POLICY[name] if mode=='hybrid' else mode)
        self.mode = mode

    def leaves(self):
        return {n:m for n,m in self.layers.named_modules() if isinstance(m,FineIntegerConv)}

    def forward(self,x): return self.layers(x)


class FineIntegerDetector(ProtectedMultiscaleRefreshDetector):
    def __init__(self, source, calibration, mode, capture_audit=False):
        nn.Module.__init__(self)
        if type(source) is not ProtectedMultiscaleRefreshDetector: raise ValueError('Protected source required')
        self.core = copy.deepcopy(source.core); self.pillar_vfe, self.scatter = source.pillar_vfe, source.scatter
        for key in ('bounds','period','max_age','light_observations','memory_policy','spatial_only'): setattr(self,key,copy.deepcopy(getattr(source,key)))
        self.core.observation = FineIntegerEncoder(source.core.observation, calibration, mode)
        self.variant = 'fine_integer_'+mode
        for leaf in self.core.observation.leaves().values(): leaf.capture_audit = capture_audit
        self.eval()

    def step_spatial(self, spatial, state, meta, detach_state=True, requests=None):
        if requests is not None or not detach_state: raise ValueError('Frozen whole stream only')
        reason,_ = reset_reason(state,meta); leaves = self.core.observation.leaves()
        previous = {} if reason else state['fine_integer']
        for name, leaf in leaves.items(): leaf.previous = previous.get(name)
        try:
            pred,new,info = super().step_spatial(spatial,state,meta,detach_state=True)
            new['fine_integer'] = {n:m.next_state for n,m in leaves.items() if m.next_state is not None}
            info['fine_integer'] = {n:dict(m.statistics) for n,m in leaves.items()}
            info['cache_bytes'] += sum(m.statistics['reference_bytes']+m.statistics['accumulator_bytes'] for m in leaves.values())
            info['audit_scratch_bytes'] = sum(v.numel()*v.element_size() for m in leaves.values() if m.audit is not None for v in m.audit)
            return pred,new,info
        finally:
            for leaf in leaves.values(): leaf.previous=None; leaf.next_state=None

    def clear_audit(self):
        for leaf in self.core.observation.leaves().values(): leaf.audit=None
