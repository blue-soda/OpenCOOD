"""Check the spatial ANN against direct convolutions and the temporal ANN against its sending rule."""
import argparse
from pathlib import Path

import torch

from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic
from opencood.tools.train_int_full_dair_pair import full_rows


@torch.no_grad()
def check(a):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    cfg = fit.load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    source = fit.build_source(a.root/'inputs',cfg)
    rows = periodic(full_rows(a.root/'sequence_manifest.json', a.dataset, 'train')[:10])
    net = RecoveryPyramid(source.core,a.root/'initial.pt','fine_temporal').cuda().eval()
    samples = fit.loader(rows,cfg,0,303)[1]
    inputs = [fit.spatial_input(source,fit.to_device(s['ego'],torch.device('cuda'))) for s in samples]
    for sender in net.senders:
        sender.full_amplitude = True
    errors = []
    for spatial in inputs:
        pred,_,_ = net(spatial,source.core,None)
        x = spatial
        branches = []
        for stage,up in zip(net.stages,source.core.deblocks):
            for block in stage:
                shortcut = x if block.skip is None else block.skip(x)
                x = (block.conv2(block.conv1(x).relu()) + shortcut).relu()
            branches.append(up(x))
        features = source.core.readout(torch.cat(branches,dim=1))
        for key,head in [('psm','cls_head'),('rm','reg_head'),('dm','dir_head')]:
            if key in pred:
                expected = net.heads[head](features)
                errors.append(float((pred[key]-expected).abs().max()))
                assert torch.allclose(pred[key],expected,rtol=1e-5,atol=1e-5)
    observed = {}
    def check_send(name):
        def hook(sender, args, output):
            target, previous = args
            delta = target - previous
            q = sender.step()
            active = delta.abs() > .5*q
            direct = torch.where(active,delta.clamp(-sender.cap*q,sender.cap*q),torch.zeros_like(target))
            actual = output[0]
            assert torch.allclose(actual,direct,rtol=1e-5,atol=1e-5)
            levels = actual/q
            observed[name] += int(((levels-levels.round()).abs() > 1e-3).sum())
        return hook
    for name,sender in net.named_modules():
        if isinstance(sender,fit.Sender):
            sender.full_amplitude = False
            sender.continuous_threshold = .5
            observed[name] = 0
            sender.register_forward_hook(check_send(name))
    state = None
    for row,spatial in zip(rows,inputs):
        if row['reset_before']:
            state = None
        _,state,_ = net(spatial,source.core,state,ordinary=not row['reset_before'])
    assert all(observed.values())
    fit.write_json(a.out,dict(frames=[r['frame'] for r in rows],maximum_spatial_head_error=max(errors),
                              fractional_continuous_values_by_sender=observed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','dataset','out'):
        parser.add_argument(name,type=Path)
    check(parser.parse_args())
