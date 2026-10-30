"""Identity-initialized, capacity-matched local/message-conditioned gain."""
import torch
from torch import nn


class SnapshotContextGain(nn.Module):
    def __init__(self, condition, channels=128, hidden=16):
        super().__init__()
        if condition not in ('local', 'message'):
            raise ValueError('Expected local control or message condition')
        self.condition = condition
        self.net = nn.Sequential(nn.Conv3d(2 * channels, hidden, 1), nn.ReLU(),
                                 nn.Conv3d(hidden, channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, membrane, decoded):
        if membrane.shape != decoded.shape or membrane.ndim != 5:
            raise ValueError('Expected matching local and decoded 3D volumes')
        other = membrane if self.condition == 'local' else decoded
        return 2 * torch.sigmoid(self.net(torch.cat((membrane, other), dim=1)))
