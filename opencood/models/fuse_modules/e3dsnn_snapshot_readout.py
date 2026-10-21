"""ANN snapshot readout in the original three unary-level addition order."""
from dataclasses import replace
import torch
from opencood.models.fuse_modules.e3dsnn_content_receiver import ContentAccumulator
from opencood.models.fuse_modules.e3dsnn_readout import residual_readout


def ann_readout(model, ego, counts, scales, contract, pose, events=None):
    basis = model.content.basis_digest()
    native = replace(contract, channel_basis='content32-'+basis)
    receiver = ContentAccumulator(basis, 'ann2bit')
    state = receiver.begin(counts, native, pose, scales)
    if events is not None and len(events) != 3:
        raise ValueError('Exactly three original unary levels required')
    for k in (1, 2, 3):
        event = (counts >= k).to(ego.dtype) if events is None else events[k-1]
        if not torch.equal(event.detach(), (counts.detach() >= k).to(ego.dtype)):
            raise ValueError('Surrogate must preserve exact original unary values')
        latent, coverage = receiver.receive(state, event, native, pose, scales, k)
    nonempty = (state.source_counts.detach().abs().sum() > 0).to(ego.dtype)
    mask = (coverage*nonempty).flatten(1, 2);supported = mask.amax(1, keepdim=True)
    local = ego.flatten(1, 2);remote = model.content.decode(latent).flatten(1, 2)
    delta = model.fusion_net(torch.cat((local, remote, mask), 1))
    residual = model.fusion_scale*supported*delta
    fused = residual_readout(local, residual, supported, model.fusion_activation, model.fusion_readout)
    fused = torch.where(supported > 0, fused, local)
    receiver.finish(state)
    return latent, fused
