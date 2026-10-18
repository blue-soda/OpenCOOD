"""Synthetic GPU dispatch screen; excludes encoder, gate, warp and detection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import ResidualStateUpdate
from opencood.models.sub_modules.int_cache_repair import refresh_tiles,expand_tiles
from opencood.tools.run_int_ego import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--layout',choices=['merged','fixed'],default='merged');p.add_argument('--crop-batch',type=int,default=8)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(303);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    module=ResidualStateUpdate().cuda().eval();x=torch.randn(1,513,100,252,device='cuda');x[:,-1:]=1
    old=x[:,256:512];records=[]
    with torch.no_grad():
        for tile in [8,16,32]:
            h,w=(100+tile-1)//tile,(252+tile-1)//tile
            for pattern in ['cluster','scattered']:
                order=torch.randperm(h*w,device='cuda') if pattern=='scattered' else torch.arange(h*w,device='cuda')
                for fraction in [.1,.25,.5]:
                    active=torch.zeros(1,1,h,w,device='cuda',dtype=torch.bool)
                    active.view(-1)[order[:max(1,round(h*w*fraction))]]=True
                    mask=expand_tiles(active,tile,(100,252))
                    for execution in ['dense','masked_dense','sparse']:
                        def run():
                            if execution=='dense':return module(x),None,dict(executed_input_area_ratio=1.,actual_write_fraction=1.)
                            return refresh_tiles(module,x,old,mask,1,2,tile,execution=execution,layout=a.layout,batch_size=a.crop_batch)
                        for _ in range(5):run()
                        times=[]
                        for _ in range(a.repeats):
                            torch.cuda.synchronize();tick=time.perf_counter();_,_,stats=run();torch.cuda.synchronize()
                            times.append(1000*(time.perf_counter()-tick))
                        row=dict(tile=tile,pattern=pattern,requested_tile_fraction=fraction,execution=execution,
                            mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),stats=stats)
                        records.append(row);print(json.dumps(row),flush=True)
    write_json(out/'report.json',dict(status='COMPLETED',repeats=a.repeats,seed=303,device=torch.cuda.get_device_name(),
        layout=a.layout,crop_batch=a.crop_batch,
        packed_shape=list(x.shape),records=records,scope='Synthetic update-module execution including crop planning/dispatch, no inference accuracy',
        caveats=['Random weights/inputs and synthetic masks, not trained detector latency.',
                 'Excludes encoder, selector, history warp, cache bookkeeping and readout/head.',
                 'Screen fragmented crop overhead only; actual sequence paired measurements remain required.']))


if __name__=='__main__':main()
