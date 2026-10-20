"""Small tile scorer: predict high-level state change without executing U."""
import torch
from torch import nn
import torch.nn.functional as F


class LocalChangeSelector(nn.Module):
    version = 'local_change_tile_mlp_v1'

    def __init__(self):
        super().__init__()
        # E, last-written E, squared E difference, aligned H, coverage and age.
        self.net = nn.Sequential(nn.Conv2d(642, 32, 1), nn.ReLU(),
                                 nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.net[-1].bias, -3.)

    def forward(self, evidence, reference, old, coverage, age, tile):
        pool = lambda x: F.avg_pool2d(x, tile, tile, ceil_mode=True, count_include_pad=False)
        descriptor = torch.cat([
            pool(evidence), pool(reference), pool((evidence-reference).square()),
            pool(old), pool(coverage.to(evidence.dtype)), pool(age)], 1)
        # Fixed transform, no running statistics or validation calibration.
        descriptor = descriptor.sign() * descriptor.abs().log1p()
        return self.net(descriptor)


def selector_state(checkpoint, source_sha, tile, fraction=None):
    """Validate deployment provenance before accepting trained scorer weights."""
    if checkpoint['version'] != LocalChangeSelector.version or checkpoint['source_sha256'] != source_sha:
        raise ValueError('Selector/source mismatch')
    if checkpoint['split'] != 'train' or checkpoint['tile'] != tile:
        raise ValueError('Selector split/tile mismatch')
    if fraction is not None and checkpoint['fraction'] != fraction:
        raise ValueError('Selector was trained for another update budget')
    return checkpoint['selector']
