"""Source-only spatial/global IF drive modulation; not calibrated confidence."""
import hashlib
import json
import torch
from torch import nn


class SnapshotSpatialDrive(nn.Module):
    def __init__(self, condition, channels=128, hidden=16):
        super().__init__()
        if condition not in ('spatial', 'global'):
            raise ValueError('Expected spatial or global drive condition')
        self.condition = condition
        self.channels = channels
        # Match initialization and leave augmentation/dropout RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260929)
            self.net = nn.Sequential(nn.Conv3d(channels, hidden, 3, padding=1),
                                     nn.ReLU(), nn.Conv3d(hidden, 1, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, drive):
        if (features.ndim != 5 or drive.ndim != 5 or features.shape[0] != 1
                or drive.shape[0] != 1 or features.shape[1] != self.channels
                or drive.shape[1] != 32 or features.shape[2:] != drive.shape[2:]):
            raise ValueError('Expected one source volume and matching 32-channel drive')
        score = self.net(features)
        if self.condition == 'global':
            score = score.mean(dim=(2, 3, 4), keepdim=True)
        # Original zero drive remains zero. IF clamp and actual byte selection
        # are applied downstream, with their existing fixed-frontier STE.
        return drive * (2 * torch.sigmoid(score))

    def basis_digest(self, parent_basis):
        spec = dict(schema='source-spatial-if-drive-v1', condition=self.condition,
                    parent_basis=parent_basis)
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode('ascii'))
        for name, value in sorted(self.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(name.encode('ascii'))
            digest.update(str((tuple(value.shape), value.dtype)).encode('ascii'))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()
