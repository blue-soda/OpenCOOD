"""Feed the phase-shared 16 MiB receiver outputs through all later layers and score the original detector."""
import json
import runpy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from opencood.tools import audit_int_spike_pyramid_ops as audit
from opencood.tools.launch_int_memory import digest


plan_file,kind = Path(sys.argv.pop(1)),sys.argv.pop(1)
plan = json.loads(plan_file.read_text())['models'][kind]
parent = Path(sys.argv[sys.argv.index('--full-parent')+1])
assert digest(parent) == plan['parent']


class FoldedLedger(audit.ConvLedger):
    def __init__(self, net, count_digits=False):
        super().__init__(net,count_digits)
        self.names = {conv:n for n,conv in net.named_modules() if conv in self.inputs}
        self.kernels = {}
        self.original = self.receive

    def receive(self, conv, events, first):
        name = self.names[conv]
        option = plan['budgets']['16']
        sender = self.inputs[conv][1]
        scale = sender.step()
        if kind == 'qat':
            scale = scale * (sender.cap / 127.)
        codes = (events / scale).round()
        if name not in self.kernels:
            assert first and sender.step_multiplier == 1
            base = option['base'][name]
            amounts = [base] + option['extra'][name]
            unit = conv.weight * scale.view(1,-1,1,1)
            self.kernels[name] = (base, [(k,unit*k) for k in amounts])
        base,entries = self.kernels[name]
        effective = codes * sender.step_multiplier
        magnitude = effective.abs()
        remaining = effective
        address = (conv.stride,conv.padding,conv.dilation,conv.groups)
        terms = []
        for k,weight in entries:
            selected = magnitude == k
            signs = effective.sign().masked_fill(~selected,0)
            terms.append(F.conv2d(signs,weight,None,*address))
            remaining = remaining.masked_fill(selected,0)
        result = F.conv2d(remaining/base,entries[0][1],conv.bias if first else None,*address)
        for term in terms:
            result = result + term
        return result


if __name__ == '__main__':
    audit.ConvLedger = FoldedLedger
    runpy.run_module('opencood.tools.select_int_stream_variant',run_name='__main__')
