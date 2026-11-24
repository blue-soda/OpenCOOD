"""Score the original detector using a phase-shared receiver at a fixed table budget."""
import json
import argparse
import runpy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from opencood.tools import audit_int_spike_pyramid_ops as audit
from opencood.tools.launch_int_memory import digest


class FoldedLedger(audit.ConvLedger):
    def __init__(self, net, count_digits, option, kind, fold_phase='both'):
        super().__init__(net,count_digits)
        self.names = {conv:n for n,conv in net.named_modules() if conv in self.inputs}
        self.option = option
        self.kind = kind
        self.fold_phase = fold_phase
        assert set(self.names.values()) == set(option['base']) == set(option['extra'])
        self.hits = torch.zeros((len(self.totals),2),dtype=torch.float64,device='cuda')
        self.misses = torch.zeros_like(self.hits)
        self.kernels = {}
        self.direct = self.original
        self.original = self.receive

    def receive(self, conv, events, first):
        name = self.names[conv]
        option = self.option
        stage,sender = self.inputs[conv]
        scale = sender.step()
        if self.kind == 'qat':
            assert sender.payload_dtype == 'int8'
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
        taps = self.taps[(conv,*events.shape[-2:])]
        branches = conv.out_channels // conv.groups
        phase = 0 if first else 1
        if self.fold_phase != 'both' and self.fold_phase != ('refresh' if first else 'ordinary'):
            self.misses[stage,phase] += (taps * codes.ne(0).sum(1)).sum(dtype=torch.float64) * branches
            return self.direct(conv, events, first)
        magnitude = effective.abs()
        remaining = effective
        address = (conv.stride,conv.padding,conv.dilation,conv.groups)
        terms = []
        for k,weight in entries:
            selected = magnitude == k
            signs = effective.sign().masked_fill(~selected,0)
            terms.append(F.conv2d(signs,weight,None,*address))
            remaining = remaining.masked_fill(selected,0)
        missed = remaining.ne(0)
        matched = magnitude.ne(0) & ~missed
        self.hits[stage,phase] += (taps * matched.sum(1)).sum(dtype=torch.float64) * branches
        self.misses[stage,phase] += (taps * missed.sum(1)).sum(dtype=torch.float64) * branches
        result = F.conv2d(remaining/base,entries[0][1],conv.bias if first else None,*address)
        for term in terms:
            result = result + term
        return result


    def receipt(self, report):
        counted = self.hits.sum(1) + self.misses.sum(1)
        visits = torch.stack([self.totals[s][1] for s in range(len(self.totals))])
        assert torch.equal(counted,visits)
        used = sum(w.numel()*w.element_size() for _,entries in self.kernels.values() for _,w in entries[1:])
        assert used == self.option['extra_bytes']
        scans = report['metrics']['scans']
        hit_g = self.hits.sum().item() / (scans*1e9)
        return dict(hits_by_stage_phase=self.hits.long().tolist(), misses_by_stage_phase=self.misses.long().tolist(),
                    extra_fp32_bytes=used, scans=scans, covered_g=report['full_cost_g']-hit_g)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('layout',type=Path)
    parser.add_argument('kind',choices=('native','qat'))
    parser.add_argument('--extra-mib', type=int, choices=(0,4,16,64), default=16,
                        help='Budget for extra folded kernels, in addition to the resident kernel per receiver.')
    parser.add_argument('--target-sha',help='Use the specified frozen target with a transferred layout.')
    parser.add_argument('--fold-phase', choices=('both','refresh','ordinary'), default='both',
                        help='Apply the table in the selected phase; use the original convolution elsewhere.')
    args,rest = parser.parse_known_args()
    model_plan = json.loads(args.layout.read_text())['models'][args.kind]
    option = model_plan['budgets'][str(args.extra_mib)]
    assert 0 <= option['extra_bytes'] <= args.extra_mib * 2**20
    parent = Path(rest[rest.index('--full-parent')+1])
    parent_sha = digest(parent)
    assert parent_sha == (args.target_sha or model_plan['parent'])
    out = Path(rest[2])
    created = []
    def build_ledger(net,count_digits=False):
        ledger = FoldedLedger(net,count_digits,option,args.kind,args.fold_phase)
        created.append(ledger)
        return ledger
    audit.ConvLedger = build_ledger
    sys.argv[1:] = rest
    runpy.run_module('opencood.tools.select_int_stream_variant',run_name='__main__')
    ledger, = created
    report = json.loads((out/'result.json').read_text())
    assert report['parent_sha'] == parent_sha
    measured = ledger.receipt(report)
    measured['fold_phase'] = args.fold_phase
    measured['extra_budget_mib'] = args.extra_mib
    measured.update(layout_sha=digest(args.layout),layout_parent=model_plan['parent'],parent_sha=parent_sha,
                    scope='Actual code stream in recursive dense plane emulation; conditional arithmetic, no hardware timing.')
    with (out/'receiver_execution.json').open('x') as stream:
        json.dump(measured,stream,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
