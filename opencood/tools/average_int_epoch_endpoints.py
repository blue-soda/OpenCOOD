"""Construct a fixed within-trajectory average, retaining the two unmodified endpoints."""
import argparse
from pathlib import Path

import torch

from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import write_json


def run(first, last, out):
    early, late = [torch.load(p/'final.pt', map_location='cpu') for p in (first,last)]
    first_contract, last_contract = early['contract'], late['contract']
    if first_contract['epochs'] != 1 or last_contract['epochs'] != 2:
        raise ValueError('Expected endpoints at the first and second epoch.')
    if dict(first_contract, epochs=2) != last_contract:
        raise ValueError('Endpoint contracts differ beyond epoch count.')
    trace = Path('epoch_01/updates.jsonl')
    if digest(first/trace) != digest(last/trace):
        raise ValueError('First-epoch optimizer traces differ.')
    before, after = early['net'], late['net']
    if before.keys() != after.keys():
        raise ValueError('Incompatible checkpoint keys.')
    average = {}
    for key, v in before.items():
        w = after[key]
        if v.shape != w.shape or v.dtype != w.dtype:
            raise ValueError(key)
        if v.is_floating_point():
            average[key] = torch.lerp(v, w, 0.5)
            if not torch.isfinite(average[key]).all():
                raise ValueError(key)
        elif torch.equal(v,w):
            average[key] = v
        else:
            raise ValueError('Non-floating buffer changed: '+key)
    provenance = dict(first_sha256=digest(first/'final.pt'), last_sha256=digest(last/'final.pt'),
                      first_trace_sha256=digest(first/trace), coefficients=[0.5,0.5])
    out.mkdir(exist_ok=False)
    torch.save(dict(net=average,contract=last_contract, averaging=provenance),out/'mean.pt')
    write_json(out/'receipt.json',dict(provenance, output_sha256=digest(out/'mean.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('first','last','out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.first,args.last,args.out)
