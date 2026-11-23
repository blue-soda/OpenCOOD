"""Paired board energy and time for dense vs. static CSR on identical cached BEV fusion inputs."""
import argparse
import copy
import os
from pathlib import Path

import torch
from torch import nn

from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.measure_int_deep_phase_energy import BoardEnergy, interval
from opencood.tools.select_int_stream_variant import relocated, sparse_fusion_conv
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic


class StaticConv(nn.Module):
    def __init__(self, layer):
        super().__init__()
        assert layer.groups == 1 and layer.stride == layer.dilation == (1,1)
        self.layer = layer

    def forward(self, x):
        y = sparse_fusion_conv(self.layer, x)
        return y if self.layer.bias is None else y + self.layer.bias.view(1,-1,1,1)


@torch.no_grad()
def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    exclusive(uuid)
    plan_file = a.root/'split/plan.json'
    assert fit.digest(plan_file) == fit.read(a.root/'split/report.json')['plan']
    plan = fit.read(plan_file)
    original = plan['rows']['internal']
    scenes = list(dict.fromkeys(r['scene'] for r in original))[:4]
    rows = []
    for scene in scenes:
        rows.extend(periodic([r for r in original if r['scene']==scene][:5]))
    rows = relocated(rows, plan, a.dataset)
    assert len(rows) == 20
    cfg = fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(a.root/'inputs', cfg)
    checkpoint = a.root/'fine_temporal_303.pt'
    assert fit.digest(checkpoint) == 'e8ae1933aaa9da4103f0dab1cc0eca89eaa08d6e3bf43fe35eeea20c59f3352b'
    net = RecoveryPyramid(source.core,a.root/'initial.pt','fine_temporal').cuda().eval()
    net.load_state_dict(torch.load(checkpoint,map_location='cpu')['net'])
    inputs = []
    dense = source.core.readout
    def capture(_, args):
        inputs.append(args[0].detach().clone())
    handle = dense.register_forward_pre_hook(capture)
    _, samples = fit.loader(rows,cfg,0,303)
    state = None
    for row, sample in zip(rows,samples):
        assert row['frame'] == sample['meta']['frame']
        if row['reset_before']:
            state = None
        ego = fit.to_device(sample['ego'],torch.device('cuda'))
        _, state, _ = net(fit.spatial_input(source,ego),source.core,state,ordinary=not row['reset_before'])
    handle.remove()
    assert len(inputs) == len(rows)
    sparse = copy.deepcopy(dense)
    for parent in list(sparse.modules()):
        for name, layer in list(parent.named_children()):
            if isinstance(layer,nn.Conv2d):
                setattr(parent,name,StaticConv(layer))
    errors = []
    for x in inputs:
        y, ref = sparse(x), dense(x)
        errors.append(float((y-ref).abs().max()))
        assert torch.allclose(y,ref,atol=5e-4,rtol=2e-4), errors[-1]
    def replay(mod, passes):
        for _ in range(passes):
            for x in inputs:
                mod(x)
        torch.cuda.synchronize()
    meter = BoardEnergy(uuid)
    results = []
    arms = dict(dense=dense, static_csr=sparse)
    for order in [('dense','static_csr'),('static_csr','dense')]*2:
        for name in order:
            exclusive(uuid)
            replay(arms[name], 1)
            start = meter.mark()
            replay(arms[name], 5)
            end = meter.mark()
            exclusive(uuid)
            energy, seconds = interval(start, end)
            measured = len(inputs)*5
            result = dict(arm=name, scans=measured, start=start, end=end, board_j_per_scan=energy/measured,
                          wall_ms_per_scan=seconds*1000/measured)
            fit.record(a.out/'intervals.jsonl',result)
            results.append(result)
    fit.write_json(a.out/'result.json',dict(uuid=uuid,gpu=torch.cuda.get_device_name(),scenes=scenes,
        frames=[r['frame'] for r in rows], checkpoint_sha=fit.digest(checkpoint), max_abs_error=max(errors),
        passes=5, intervals=results, scope='Full fusion readout only. Same 20 GPU-resident inputs, '
        'FP32, includes CSR construction for every conv, synchronized wall time and gross board joules.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','dataset','out'):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
