"""Read-only diagnostics for event fanout and temporal pointwise changes.

Dense audit operations and shadow activations are not deployment savings.
"""
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_fine_integer import FineIntegerConv


def output_support(delta, conv):
    if delta.ndim != 4 or delta.shape[0] != 1 or delta.shape[1] != conv.in_channels:
        raise ValueError('Single stream with matching input channels required')
    if conv.groups != 1 or conv.dilation != (1, 1) or conv.padding_mode != 'zeros':
        raise ValueError('Ungrouped undilated zero padding required')
    active = (delta != 0).any(dim=1, keepdim=True).float()
    return F.max_pool2d(active, conv.kernel_size, conv.stride, conv.padding) != 0


class FineSupportAudit:
    def __init__(self, encoder):
        self.previous = {}; self.current = {}; self.records = {}; self.calls = {}
        self.handles = []
        for name, module in encoder.layers.named_modules():
            if isinstance(module, (FineIntegerConv, nn.BatchNorm2d, nn.ReLU)):
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def begin(self, reset):
        if reset: self.previous = {}
        self.current = {}; self.records = {}; self.calls = {}

    def hook(self, name):
        def observe(module, inputs, output):
            call = self.calls.get(name, 0); self.calls[name] = call + 1
            key = name + ':' + str(call); old = self.previous.get(key)
            # Clone before an in-place ReLU can change a BN hook's value.
            self.current[key] = output.detach().clone()
            changed = torch.ones_like(output, dtype=torch.bool) if old is None else output != old
            row = dict(kind=type(module).__name__,elements=output.numel(),spatial_positions=output.shape[-2]*output.shape[-1],
                       cold=old is None,temporal_changed_elements=int(changed.sum()),
                       temporal_changed_positions=int(changed.any(dim=1).sum()),shadow_bytes=output.numel()*output.element_size())
            if isinstance(module,FineIntegerConv):
                if module.audit is None: raise ValueError('Fine n/A audit capture required')
                n, acc = module.audit
                delta_mode = module.mode == 'delta' and module.previous is not None
                delta = n-module.previous[0].to(torch.int64) if delta_mode else n
                update = acc-module.previous[1].to(torch.int64) if delta_mode else acc
                support = output_support(delta,module.source)
                actual = update != 0
                if bool((actual & ~support).any()): raise RuntimeError('Output outside structural support')
                row.update(mode=module.mode,delta_update=delta_mode,input_elements=n.numel(),input_changed_elements=int((delta!=0).sum()),
                           candidate_positions=int(support.sum()),candidate_elements=int(support.sum())*acc.shape[1],
                           nonzero_update_elements=int(actual.sum()),nonzero_update_positions=int(actual.any(dim=1).sum()),
                           initial_pointwise_required_elements=output.numel() if old is None else None)
                if delta_mode and old is not None and bool((changed & ~actual).any()):
                    raise RuntimeError('Decoded convolution changed without integer accumulator update')
            self.records[key] = row
        return observe

    def end(self):
        previous_bytes=sum(v.numel()*v.element_size() for v in self.previous.values())
        current_bytes=sum(v.numel()*v.element_size() for v in self.current.values())
        result=dict(modules=self.records,previous_shadow_bytes=previous_bytes,current_shadow_bytes=current_bytes,
                    simultaneous_shadow_bytes=previous_bytes+current_bytes)
        self.previous=self.current;self.current={}
        return result

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.previous={};self.current={}
