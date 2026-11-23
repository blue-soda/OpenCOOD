"""Count theoretical event synapses on the scored dense QAT backbone trace."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from opencood.tools import train_int_spike_pyramid as fit


def signed_powers(value):
    """Non-adjacent signed binary digits, returned as signed powers of two."""
    terms = []
    power = 1
    while value:
        if value % 2:
            digit = 2 - value % 4
            terms.append(digit * power)
            value -= digit
        value //= 2
        power *= 2
    return terms


class ConvLedger:
    def __init__(self, net, count_digits=False):
        src = net.input_sender
        self.inputs, self.taps, self.totals = {}, {}, {}
        for s, stage in enumerate(net.stages):
            self.totals[s] = torch.zeros(4, device='cuda', dtype=torch.float64)
            for block in stage:
                self.inputs[block.conv1] = (s,src)
                if block.skip is not None:
                    self.inputs[block.skip] = (s,src)
                self.inputs[block.conv2] = (s,block.sender1)
                src = block.sender2
        self.original = fit.delta_conv
        self.digit_visits = {}
        if count_digits:
            terms = [signed_powers(i) for i in range(128)]
            assert all(sum(row) == i for i,row in enumerate(terms))
            self.digit_counts = torch.tensor([[bin(i).count('1'),len(row)] for i,row in enumerate(terms)],
                                             device='cuda',dtype=torch.float32)
            self.digit_visits = {stage:torch.zeros(4,device='cuda',dtype=torch.float64)
                                 for stage in self.totals}

    def count(self, conv, events, first):
        stage, sender = self.inputs[conv]
        q = sender.step()
        raw = events / q
        continuous = sender.continuous_threshold is not None or sender.full_amplitude
        if not continuous:
            code = raw.round()
            assert (raw - code).abs().max() < 1e-4 and code.abs().max() <= sender.cap
        h,w = events.shape[-2:]
        key = (conv,h,w)
        if key not in self.taps:
            kernel = events.new_ones(1,1,*conv.kernel_size)
            # Gradient of the sum of all valid outputs counts the fan-out at each input.
            out = F.conv2d(events.new_ones(1,1,h,w), kernel, stride=conv.stride,
                           padding=conv.padding, dilation=conv.dilation)
            self.taps[key] = torch.nn.grad.conv2d_input((1,1,h,w),kernel,torch.ones_like(out),
                stride=conv.stride,padding=conv.padding,dilation=conv.dilation)[0,0]
        taps = self.taps[key]
        branches = conv.out_channels // conv.groups
        if self.digit_visits:
            if sender.payload_dtype == 'int8':
                levels = events / (q * (sender.cap / 127.))
                magnitude = levels.round().abs()
                assert (levels-levels.round()).abs().max() < 1e-4 and magnitude.max() <= 127
            elif not continuous:
                magnitude = code.abs()
            else:
                raise ValueError('Digit costs require integer codes at all backbone senders.')
            digits = self.digit_counts[magnitude.long()].sum(1)
            visits = torch.stack(((taps * magnitude.ne(0).sum(1)).sum(dtype=torch.float64),
                                  (taps * magnitude.sum(1)).sum(dtype=torch.float64),
                                  (taps * digits[...,0]).sum(dtype=torch.float64),
                                  (taps * digits[...,1]).sum(dtype=torch.float64)))
            self.digit_visits[stage] += visits * branches
        if continuous:
            self.totals[stage][0] += taps.sum(dtype=torch.float64) * conv.in_channels * branches
            self.totals[stage][1] += (taps * events.ne(0).sum(1)).sum(dtype=torch.float64) * branches
            # Repeated unit-spike costs do not apply to continuous amplitudes.
            self.totals[stage][2:] = float('nan')
            return self.original(conv,events,first)
        counts = torch.stack((taps.sum(dtype=torch.float64)*conv.in_channels,
            (taps * (code != 0).sum(1)).sum(dtype=torch.float64),
            (taps * code.abs().sum(1)).sum(dtype=torch.float64),
            (taps * code.abs().clamp_max(2).sum(1)).sum(dtype=torch.float64)))
        self.totals[stage] += counts * branches
        return self.original(conv,events,first)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = fit.load_yaml(str(a.inputs/'config.yaml'))
    cfg['data_augment'] = []
    contract = fit.read(a.run/'contract.json')
    assert fit.digest(a.plan) == contract['plan'] and (a.run/'report.json').is_file()
    scenes = contract['monitor']
    rows = [r for r in fit.read(a.plan)['rows']['fit'] if r['scene'] in scenes]
    interval = 1 if a.reset_each_scan else a.reset_interval
    anchor_mode = a.anchor_mode or contract.get('anchor_mode')
    clear_history = anchor_mode and (a.reset_each_scan or contract.get('memoryless', False))
    if contract.get('memoryless') and interval is None:
        interval = 1
    if interval is not None and not anchor_mode:
        if interval < 1:
            raise ValueError('reset interval must be positive')
        phase = 0
        scheduled = []
        for i, row in enumerate(rows):
            if i == 0 or row['segment'] != rows[i-1]['segment'] or row.get('reset_before'):
                phase = 0
            scheduled.append(dict(row, reset_before=phase == 0))
            phase = (phase + 1) % interval
        rows = scheduled
    source = fit.build_source(a.inputs,cfg)
    net = fit.SpikePyramid(source.core,contract['cap'],contract.get('stage_caps')).cuda()
    weights = torch.load(a.run/'best.pt',map_location='cpu')
    net.load_state_dict(weights['net'])
    ledger = ConvLedger(net)
    fit.delta_conv = ledger.count
    try:
        validation = fit.replay(rows,cfg,source,net,fit.PointPillarDirLoss(cfg['loss']['args']).cuda(),
                                None,weights['epoch'],a.run,{},memoryless=bool(clear_history),
                                anchor_mode=anchor_mode)
    finally:
        fit.delta_conv = ledger.original
    if anchor_mode == contract.get('anchor_mode') and (interval is None or (contract.get('memoryless') and interval == 1)):
        assert abs(validation['ap70'] - weights['best']) < 1e-6
    fit.write_json(a.output,dict(checkpoint=fit.digest(a.run/'best.pt'),epoch=weights['epoch'],
        code=fit.digest(Path(__file__)), monitor=scenes, validation=validation,
        reset_each_scan=interval==1, reset_interval=interval,
        anchor_mode=anchor_mode, anchor_period=fit.TranslationAnchor.period if anchor_mode else None,
        columns=['dense_valid_MAC','same_event_nonzero_MAC','repeat_unit_AC','unit_add_else_MAC_equal_cost_ops'],
        stage_per_scan=[(v/len(rows)).tolist() for v in ledger.totals.values()],
        scope='residual convolutions including projected skips, excluding padding; unit-weight repeats; theoretical only'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    reset = p.add_mutually_exclusive_group()
    reset.add_argument('--reset-each-scan',action='store_true')
    reset.add_argument('--reset-interval',type=int)
    p.add_argument('--anchor-mode',choices=['p5','fixed','translation'])
    for name in ('inputs','plan','run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    if args.anchor_mode and args.reset_interval:
        p.error('anchor-mode already defines the five-scan reset interval')
    run(args)
