"""Explicit weights-only initialization, never mistaken for optimizer resume."""
import torch
from opencood.tools.launch_int_memory import digest


def load_warm_start(repair,path,contract):
    saved=torch.load(str(path),map_location='cpu')
    previous=saved['contract']
    assert previous.get('kind')==contract.get('kind')=='current'
    for key in ('source','frozen_deep_checkpoint','width','deep_period','trainable_parameters','architecture'):
        assert previous[key]==contract[key], 'Warm-start identity mismatch: '+key
    assert previous['readout_period'] in (3,5) and contract['readout_period'] in (3,5)
    repair.load_state_dict(saved['repair'],strict=True)
    return dict(path=str(path),sha256=digest(path),epoch=saved['epoch'],
                source_period=previous['readout_period'],target_period=contract['readout_period'],
                optimizer='Fresh AdamW and plateau schedule; no optimizer, scheduler or epoch history restored')
