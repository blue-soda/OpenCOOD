"""Check signed-power residual convolutions against the scored floating-amplitude operators."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.audit_int_spike_pyramid_ops import ConvLedger, signed_powers
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid, periodic


@torch.no_grad()
def run(root, out):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    cfg = fit.load_yaml(str(root/'inputs/config.yaml'))
    cfg['data_augment'] = []
    src = fit.build_source(root/'inputs',cfg)
    rows = periodic(v2v_rows(root/'formal_splits/v2v_official_deduplicated.jsonl','train')[:2])
    assert [r['reset_before'] for r in rows] == [True,False]
    inputs = [fit.spatial_input(src, fit.to_device(item['ego'],torch.device('cuda')))
              for item in fit.loader(rows,cfg,0,303)[1]]
    lookup = torch.zeros((128,8),device='cuda')
    for i in range(128):
        for term in signed_powers(i):
            lookup[i,abs(term).bit_length()-1] = 1 if term>0 else -1
    powers = [2**j for j in range(8)]
    assert torch.equal(lookup @ lookup.new_tensor(powers), torch.arange(128,device='cuda').float())
    report = {}
    for arm, ck in (('spike','v2v_full_temporal'),('int8','continuous528_fit')):
        net = RecoveryPyramid(src.core, root/'initial.pt','fine_temporal').cuda().eval()
        net.load_state_dict(torch.load(root/ck/'final.pt',map_location='cpu')['net'])
        if arm == 'int8':
            for sender in net.senders:
                sender.continuous_threshold = .5
                sender.payload_dtype = 'int8'
        ledger = ConvLedger(net)
        observed = []

        def check(conv, events, first):
            stage,sender = ledger.inputs[conv]
            scale = sender.step()
            if arm == 'int8':
                scale = scale*(sender.cap/127.)
            levels = events/scale
            code = levels.round()
            assert (levels-code).abs().max() < 1e-4 and code.abs().max() <= 127
            reference = ledger.original(conv,events,first)
            grouped = conv.weight.reshape(conv.groups,conv.out_channels//conv.groups,
                                          conv.in_channels//conv.groups,*conv.kernel_size)
            weights = (grouped*scale.reshape(conv.groups,1,-1,1,1)).reshape_as(conv.weight)
            result = torch.zeros_like(reference)
            for j,power in enumerate(powers):
                plane = lookup[:,j][code.abs().long()]*code.sign()
                if plane.any():
                    result.add_(F.conv2d(plane,weights*power,None,conv.stride,conv.padding,
                                         conv.dilation,conv.groups))
            if first and conv.bias is not None:
                result.add_(conv.bias.reshape(1,-1,1,1))
            peak = float(reference.abs().max())
            error = float((result-reference).abs().max())
            assert error <= 2e-4*max(1.,peak)
            observed.append(dict(stage=stage, first=first, error=error, peak=peak))
            return reference

        fit.delta_conv = check
        try:
            state = None
            for row,x in zip(rows,inputs):
                _,state,_ = net(x,src.core,state,ordinary=not row['reset_before'])
        finally:
            fit.delta_conv = ledger.original
        assert len(observed) == 2*len(ledger.inputs)
        report[arm] = observed
    fit.write_json(out,dict(frames=[r['frame'] for r in rows], calls=report,
                            scope='Dense per-digit reference, one refresh and one ordinary scan; '
                                  'FP32 numerical verification only, not a sparse implementation or timing.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    run(a.root,a.out)
