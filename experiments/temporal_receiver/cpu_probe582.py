"""Run the five-scan scale check on a CPU when the validation server's GPUs are occupied."""
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import probe582


if __name__ == '__main__':
    # Historical inference helpers hard-code CUDA at model transfer, input
    # transfer and creation of ConvLedger's four unneeded per-stage counters.
    # Keep model, checkpoint loading and all arithmetic on their normal path.
    cpu = torch.device('cpu')
    zeros = torch.zeros
    transfer = probe582.fit.to_device
    def place_zeros(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = cpu
        return zeros(*args, **kwargs)
    with patch.object(torch.nn.Module,'cuda',lambda module:module.to(cpu)), \
         patch.object(torch,'zeros',place_zeros), \
         patch.object(probe582.fit,'to_device',lambda data,device:transfer(data,cpu)):
        probe582.run(*map(Path,sys.argv[1:4]),*sys.argv[4:6])
