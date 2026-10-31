"""Frozen-checkpoint replay and paired, hook-free model latency for DAIR ports.

Consumes best.pth files with their saved contracts. Does not train or select
checkpoints. Benchmark timing excludes preprocessing, H2D and postprocessing.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from opencood.data_utils.int_ego_stream import read_json, sha256
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle
from opencood.models.sub_modules.mural_resolution import install_resolution_bn, select_resolution
from opencood.tools.run_mural_dair import MultiResolutionStream, data_loader, evaluate, write
from opencood.tools.train_utils import to_device


def load_frozen(path, manifest_sha):
    saved = torch.load(path, map_location='cpu')
    contract = saved['contract']
    if contract['manifest_sha256'] != manifest_sha:
        raise ValueError('Frozen manifest mismatch')
    configs = contract['configs']
    model = PointPillarCodyntrustSingle(configs[0]['model']['args'])
    if contract['arguments']['mode'] == 'mural':
        install_resolution_bn(model, len(configs))
    model.load_state_dict(saved['model'], strict=True)
    return model.cuda().eval(), contract, saved['epoch']


def gpu_process_sample(gpu):
    uuid = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader']).decode().strip()
    lines = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).decode().splitlines()
    pids = [int(x.split(',')[1]) for x in lines if x.split(',')[0].strip() == uuid]
    return dict(monotonic=time.monotonic(),gpu_uuid=uuid,pids=pids,
                other_pids=[p for p in pids if p != os.getpid()])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',nargs='+',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['replay','benchmark'],required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--eval-limit',type=int,default=0)
    p.add_argument('--physical-gpu',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    random.seed(303);np.random.seed(303);torch.manual_seed(303);torch.cuda.manual_seed_all(303)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    manifest=read_json(args.manifest);manifest_sha=sha256(args.manifest)
    names=[Path(x).parent.name for x in args.checkpoint]
    if len(set(names))!=len(names):raise ValueError('Checkpoint parent names must be unique')
    inputs=[];loaded=[]
    for name,path in zip(names,args.checkpoint):
        model,contract,epoch=load_frozen(path,manifest_sha)
        if contract['arguments']['seed']!=303 or contract['arguments']['sweeps']!=3:
            raise ValueError('Expected frozen seed303 / 3-scan protocol')
        loaded.append((name,model,contract))
        inputs.append(dict(name=name,path=path,sha256=sha256(path),epoch=epoch,contract=contract))
    write(out/'contract.json',dict(arguments=vars(args),inputs=inputs,manifest_sha256=manifest_sha,
          code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
          timing_scope='Model forward only, synchronized, no operation-count hooks; excludes IO, alignment, voxelization, H2D and postprocess'))
    write(out/'status.json',dict(status='RUNNING',mode=args.mode))
    if args.mode=='replay':
        reports={}
        for name,model,contract in loaded:
            target=out/name;target.mkdir()
            dataset=MultiResolutionStream(manifest['val'],contract['configs'],3,303)
            reports[name]=evaluate(model,dataset,args,target)
        write(out/'reports.json',reports)
    else:
        # Construct each distinct grid once; all variants see identical per-grid input.
        configs=[];cases=[]
        for name,model,contract in loaded:
            for k,config in enumerate(contract['configs']):
                if config not in configs:configs.append(config)
                index=configs.index(config)
                cases.append((name+'/r'+str(k),model,k,config,index))
        dataset=MultiResolutionStream(manifest['val'],configs,3,303)
        values={name:[] for name,*_ in cases};samples=[];scans=0
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),(out/'frames.jsonl').open('w',buffering=1) as f:
            for sample in data_loader(dataset,args.workers):
                if args.eval_limit and scans>=args.eval_limit:break
                if scans%50==0:samples.append(gpu_process_sample(args.physical_gpu))
                times={}
                for j in range(len(cases)):
                    name,model,k,config,index=cases[(scans+j)%len(cases)]
                    ego=to_device(sample['egos'][index],torch.device('cuda'))
                    select_resolution(model,config,k)
                    torch.cuda.synchronize();start=time.perf_counter()
                    prediction=model(ego)
                    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                    if not all(torch.isfinite(v).all() for v in prediction.values()):
                        raise ValueError('Nonfinite prediction')
                    times[name]=ms
                    if scans>=20:values[name].append(ms)
                    del prediction,ego
                f.write(json.dumps(dict(frame=sample['meta']['frame'],model_ms=times))+'\n')
                scans+=1
                if scans%200==0:
                    print('BENCH',scans,flush=True);write(out/'status.json',dict(status='RUNNING',scans=scans))
        samples.append(gpu_process_sample(args.physical_gpu))
        write(out/'benchmark.json',dict(scans=scans,full_flow=scans==len(dataset),warmup_scans=20,
              timing_valid_no_observed_gpu_sharing=not any(x['other_pids'] for x in samples),
              exclusivity_scope='Compute-process samples every 50 frames and at end; not continuous enforcement',
              gpu_process_samples=samples,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_scope='All loaded model weights plus current activations; not per-model peak memory',
              results={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),n=len(v)) for k,v in values.items()}))
    write(out/'status.json',dict(status='COMPLETED',mode=args.mode))


if __name__=='__main__':main()
