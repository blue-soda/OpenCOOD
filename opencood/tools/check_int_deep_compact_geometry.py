"""Audit and time cold residual candidate compilation on recorded mask geometries."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import subprocess
import time
from types import SimpleNamespace

import torch

from opencood.models.sub_modules.int_cache_repair import tile_mask
from opencood.models.sub_modules.int_pyramid_regions import compact_tile_geometry, plan_coalesced_refresh
from opencood.tools.eval_int_deep_block_rank import CAP, TILE
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import (
    add_tiles, compact_group_bucket, group_bucket, masks_by_tile,
)
from opencood.tools.run_int_ego import write_json


PASSES = 16


def masks(compose, long, shape):
    h,w = shape
    counts = Counter()
    for line in compose.read_text().splitlines():
        row = json.loads(line)
        if row['refresh'] is None:
            counts[tuple(row['mandatory'])] += 1
    for line in long.read_text().splitlines():
        row = json.loads(line)
        if row['refresh'] is None:
            written = row.get('written_indices',row['written'])
            assert isinstance(written,list)
            picked = row['arms']['indexed_crop_online']['selection']['chosen'] or []
            optional = {y*w+x for ty,tx in picked for y in range(ty*TILE,min((ty+1)*TILE,h))
                        for x in range(tx*TILE,min((tx+1)*TILE,w))}
            assert optional.issubset(written)
            counts[tuple(sorted(set(written)-optional))] += 1
    return counts


def workloads(mask):
    permitted = set(masks_by_tile(mask,TILE))
    yield None,mask
    for y,x in sorted(permitted):
        for neighbor in ((y+1,x),(y,x+1)):
            if neighbor in permitted:
                pair = ((y,x),neighbor)
                yield pair,add_tiles(mask,pair,TILE)


def run(args,out):
    compose,long = Path(args.compose_proof),Path(args.long_proof)
    shape = tuple(args.lattice)
    input_shape = tuple(args.input_shape)
    assert shape==tuple((d+1)//2 for d in input_shape)
    counts = masks(compose,long,shape)
    assert counts
    cache = SimpleNamespace(tile=TILE,stage=SimpleNamespace(radius=args.radius))
    frame = SimpleNamespace(coarse=torch.empty(1,1,*input_shape))
    hashes = {n:digest(p) for n,p in [('compose',compose),('long',long)]}
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        inputs_sha256=hashes,torch=torch.__version__,shape=shape,input_shape=input_shape,radius=args.radius,
        cap=CAP,tile=TILE,passes=PASSES,scope='CPU only, uncached geometry compilation. Includes no feature scoring or CUDA work.'))
    results = []
    with (out/'geometry.jsonl').open('x',buffering=1) as raw:
        for indices,observations in sorted(counts.items()):
            mask = torch.zeros(1,1,*shape,dtype=torch.bool)
            mask.flatten()[list(indices)] = True
            audited = []
            for pair,requested in workloads(mask):
                original = plan_coalesced_refresh(requested,input_shape,args.radius,TILE)
                active = tile_mask(requested,TILE)[0,0].tolist()
                assert compact_tile_geometry(active,input_shape,shape,args.radius,TILE)==original[1:]
                audited.append(dict(pair=pair,work_sites=original[2],layout_size=original[4]))
            expected = group_bucket(cache,frame,mask,CAP)
            assert compact_group_bucket(cache,frame,mask,CAP)==expected
            samples = []
            modes = ('original','compact')
            compilers = dict(original=group_bucket,compact=compact_group_bucket)
            for p in range(PASSES):
                order = modes if p%2==0 else modes[::-1]
                ms = {}
                for mode in order:
                    start = time.perf_counter()
                    result = compilers[mode](cache,frame,mask,CAP)
                    ms[mode] = (time.perf_counter()-start)*1000
                    assert result==expected
                samples.append(dict(order=order,ms=ms))
            row = dict(mandatory=indices,observations=observations,audited=audited,chosen_bucket=expected,samples=samples)
            raw.write(json.dumps(row,allow_nan=False)+'\n')
            results.append(row)
    assert hashes=={'compose':digest(compose),'long':digest(long)}
    times = [p['ms'] for r in results for p in r['samples']]
    write_json(out/'report.json',dict(status='COMPLETED',unique_geometries=len(results),observed=sum(counts.values()),
        evaluated_layouts=sum(len(r['audited']) for r in results),exact=True,
        mean_cold_ms={m:mean(t[m] for t in times) for m in ('original','compact')},
        original_minus_compact_ms=mean(t['original']-t['compact'] for t in times)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('compose-proof','long-proof','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--lattice',nargs=2,type=int,required=True)
    p.add_argument('--input-shape',nargs=2,type=int,required=True)
    p.add_argument('--radius',type=int,required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise
    write_json(out/'status.json',dict(status='COMPLETED'))


if __name__ == '__main__':
    main()
