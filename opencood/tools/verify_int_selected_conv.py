"""CPU-only numerical audit at the real projected-GRU spatial dimensions.

Uses frozen trained weights with synthetic inputs, not a dataset/AP benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3


def main():
    parser=argparse.ArgumentParser()
    for key in ('checkpoint','sha256','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args();path=Path(args.checkpoint);out=Path(args.output)
    if out.exists():
        raise FileExistsError(str(out))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=args.sha256:
        raise ValueError('Checkpoint SHA mismatch')
    torch.set_num_threads(2);torch.manual_seed(303)
    saved=torch.load(str(path),map_location='cpu');results={}
    assert saved['contract']['rank']==32
    height,width=200,504
    with torch.no_grad():
        for name in ('reset_gate_conv','update_gate_conv','candidate_conv'):
            weight=saved['memory']['fusion.'+name+'.1.1.weight']
            assert tuple(weight.shape)==(64,32,3,3)
            conv=torch.nn.Conv2d(32,64,3,bias=False).eval()
            conv.weight.copy_(weight)
            current=torch.randn(1,32,height,width)
            cache=torch.randn(1,64,height,width)
            dense=conv(F.pad(current,(1,1,1,1)))
            tiles=torch.rand(1,1,height//4,width//4)<.65
            masks=dict(full=torch.ones(1,1,height,width,dtype=torch.bool),
                tiled=tiles.repeat_interleave(4,2).repeat_interleave(4,3))
            values={}
            for label,mask in masks.items():
                actual,macs=selected_conv3x3(conv,current,cache,mask)
                expected=torch.where(mask,dense,cache)
                error=float((actual-expected).abs().max())
                assert torch.isfinite(actual).all()
                assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-4),error
                assert torch.equal(actual[~mask.expand_as(actual)],cache[~mask.expand_as(cache)])
                selected=int(mask.sum());assert macs==selected*32*64*9
                values[label]=dict(max_abs_error=error,selected_outputs=selected,
                    output_fraction=selected/float(height*width),executed_conv_macs=macs)
            results[name]=values
    report=dict(status='PASSED',checkpoint=str(path),sha256=args.sha256,epoch=saved['epoch'],
        shape=[1,32,height,width],device='cpu',threads=2,results=results,
        scope='Frozen trained gate weights; synthetic inputs and full/random4x4 tile masks. Numerical equivalence and actual linear MAC accounting only. No real-data locality, GPU timing, recurrent feedback or AP claim.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
    print(json.dumps(report))


if __name__=='__main__':
    main()
