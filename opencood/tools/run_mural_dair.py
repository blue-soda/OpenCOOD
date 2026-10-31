"""Bounded, resumable DAIR migration of MURAL's multi-resolution training core.

Three historical scans are identical across modes. No future data, teacher,
deadline scheduler, forecasted boxes, or TensorRT optimization in this stage.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from opencood.data_utils.int_ego_stream import DairEgoStreamDataset, read_json, sha256, stream_collate
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.point_pillar_codyntrust_single import PointPillarCodyntrustSingle
from opencood.models.sub_modules.mural_resolution import install_resolution_bn, select_resolution, mode_configs, OperationCounter
from opencood.tools.train_utils import to_device
from opencood.utils import pcd_utils, eval_utils


def write(path, value):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


class MultiResolutionStream(Dataset):
    def __init__(self, rows, configs, sweeps=3, seed=303, labelled_only=False):
        self.rows, self.configs, self.sweeps, self.seed = rows, configs, sweeps, seed
        self.processors = [DairEgoStreamDataset(rows, c, seed) for c in configs]
        self.indices = [i for i, r in enumerate(rows) if r['supervised'] or not labelled_only]
        self.histories = []
        for i, r in enumerate(rows):
            selected = [i]
            for j in range(i-1, max(-1, i-sweeps), -1):
                if rows[j]['segment'] != r['segment']: break
                dt = (rows[j+1]['timestamp_us']-rows[j]['timestamp_us'])/1e6
                if not 0 < dt <= .25 or rows[j+1].get('reset_before', False): break
                selected.append(j)
            self.histories.append(selected)

    @lru_cache(maxsize=8)
    def points(self, path):
        p, _ = pcd_utils.read_pcd(path)
        if not len(p): raise ValueError('Empty frozen scan: '+path)
        return pcd_utils.mask_ego_points(p)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]; meta = dict(self.rows[i]); parts = []
        for j in self.histories[i]:
            r = self.rows[j]; p = self.points(r['pcd']).copy()
            if j != i:
                transform = np.linalg.solve(np.asarray(meta['pose']), np.asarray(r['pose']))
                p[:, :3] = p[:, :3].dot(transform[:3, :3].T) + transform[:3, 3]
            parts.append(p)
        points = np.concatenate(parts)
        points = pcd_utils.mask_points_by_range(points, self.configs[0]['preprocess']['cav_lidar_range'])
        points = points[np.random.RandomState(self.seed+int(meta['frame'])).permutation(len(points))]
        meta.update(history_frames=[self.rows[j]['frame'] for j in self.histories[i]], point_count=len(points))
        egos = []
        annotations = read_json(meta['label']) if meta['supervised'] else None
        for processor in self.processors:
            processed = processor.pre.collate_batch([processor.pre.preprocess(points.copy())])
            if processed['voxel_features'].shape[0] < 2: raise ValueError('Insufficient voxels')
            processed['batch_size'] = 1
            ego = dict(processed_lidar=processed, batch_size=1, anchor_box=torch.from_numpy(processor.anchors),
                       transformation_matrix=torch.eye(4), transformation_matrix_clean=torch.eye(4))
            if annotations is not None:
                boxes, mask, ids = processor.post.generate_object_center_dairv2x_single([{'params': {'vehicles': annotations}}])
                labels = processor.post.generate_label(gt_box_center=boxes, anchors=processor.anchors, mask=mask)
                ego.update(object_bbx_center=torch.from_numpy(boxes[None]).float(), object_bbx_mask=torch.from_numpy(mask[None]).float(),
                           object_ids=ids, label_dict=processor.post.collate_batch([labels]))
            egos.append(ego)
        return dict(meta=meta, egos=egos)


def data_loader(dataset, workers, shuffle=False, seed=303):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=workers, collate_fn=stream_collate,
                      generator=torch.Generator().manual_seed(seed), pin_memory=False)


def evaluate(model, dataset, args, output):
    model.eval(); counter = OperationCounter(model)
    stats = [{t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5,.7)} for _ in dataset.configs]
    costs = [[] for _ in stats]; times = [[] for _ in stats]; labels = 0; scans = 0
    with torch.no_grad(), (output/'frames.jsonl').open('w', buffering=1) as log:
        for sample in data_loader(dataset,args.workers):
            if args.eval_limit and scans >= args.eval_limit: break
            meta=sample['meta']; record=dict(frame=meta['frame'],history_frames=meta['history_frames'],point_count=meta['point_count'],modes={})
            # Rotate execution order to reduce order bias within each frame.
            for k in [(scans+j)%len(stats) for j in range(len(stats))]:
                ego=to_device(sample['egos'][k],torch.device('cuda'));select_resolution(model,dataset.configs[k],k)
                counter.reset();torch.cuda.synchronize();start=time.perf_counter();prediction=model(ego);torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000;gflops=counter.macs*2/1e9
                if not all(torch.isfinite(x).all() for x in prediction.values()):raise ValueError('Nonfinite prediction')
                costs[k].append(gflops)
                if scans>=10:times[k].append(ms)
                record['modes'][str(k)]=dict(model_ms=ms,conv_linear_gflops=gflops,voxels=int(ego['processed_lidar']['voxel_features'].shape[0]))
                if meta['supervised']:
                    post=dataset.processors[k].post;batch={'ego':ego}
                    boxes,scores=post.post_process(batch,{'ego':prediction});gt=post.generate_gt_bbx(batch)
                    for t in stats[k]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[k],t)
            labels+=int(meta['supervised']);scans+=1;log.write(json.dumps(record)+'\n')
            if scans%400==0:print('EVAL',output.name,scans,flush=True)
    counter.close()
    result=dict(scans=scans,labels=labels,full_flow=scans==len(dataset),results=[])
    result['ap_valid']=labels>0 and all(s[.7]['gt']>0 for s in stats)
    if len(set(s[.7]['gt'] for s in stats))!=1:raise ValueError('Resolution-dependent GT count')
    if result['full_flow'] and len(dataset.rows)==5794 and (labels!=1738 or stats[0][.7]['gt']!=22971):
        raise ValueError('Frozen validation coverage mismatch')
    for k,s in enumerate(stats):
        result['results'].append(dict(resolution=k,grid=dataset.configs[k]['model']['args']['point_pillar_scatter']['grid_size'],
            ap50=eval_utils.calculate_ap(s,.5)[0] if result['ap_valid'] else None,
            ap70=eval_utils.calculate_ap(s,.7)[0] if result['ap_valid'] else None,gt=s[.7]['gt'],
            conv_linear_gflops=float(np.mean(costs[k])),mean_ms=float(np.mean(times[k])) if times[k] else None,
            p95_ms=float(np.percentile(times[k],95)) if times[k] else None))
    result['timing_scope']='Model only; CUDA synchronized; excludes IO, history alignment, voxelization, H2D, postprocess; no end-to-end claim'
    write(output/'evaluation.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['fixed','mural','shared_bn'],required=True)
    p.add_argument('--fixed-resolution',type=int,choices=[0,1,2],default=0)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=303);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--sweeps',type=int,default=3);p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--eval-limit',type=int,default=0);p.add_argument('--resume');p.add_argument('--eval-only',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=bool(args.resume))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=load_yaml(args.config);configs=mode_configs(base,args.mode,args.fixed_resolution)
    manifest=read_json(args.manifest)
    model=PointPillarCodyntrustSingle(configs[0]['model']['args'])
    state=torch.load(args.checkpoint,map_location='cpu');state=state.get('model',state)
    model.load_state_dict(state,strict=True)
    if args.mode=='mural':install_resolution_bn(model,len(configs))
    model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[6,10],gamma=.3)
    start_epoch=0;history=[]
    if args.resume:
        saved=torch.load(args.resume,map_location='cuda');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_epoch=saved['epoch'];history=saved['history'];random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['rng_cuda']])
    contract=dict(arguments=vars(args),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        checkpoint_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),configs=configs,
        train_scans=len(manifest['train']),val_scans=len(manifest['val']),
        scope='MURAL multi-resolution/RABN core on DAIR with matched causal 3-scan input; NOT full author runtime reproduction',
        differences=['DAIR BEV AP instead of nuScenes mAP','OpenCOOD ResNet PointPillars and anchor head','3 sweeps instead of 10','near-square grids preserve identical field of view and divisible-by-8 geometry','P1 warm-start; 12-epoch bounded pilot','no forecasting, scheduler, BN interpolation, DCO or TensorRT yet'],
        training='All weights trainable, resolution-specific running statistics trained; sum losses across resolutions, one optimizer update per labelled scan; same update count, unequal training FLOPs',
        input='xyz/intensity only; historical ego-point mask in original coordinates; SE3 alignment; no timestamp feature, no future scans',
        stop='epoch_budget_not_convergence; latest contains full optimizer/scheduler/RNG, best is validation-selected')
    if args.resume:
        previous=read_json(out/'contract.json')
        for key in ('checkpoint_sha256','manifest_sha256','configs'):
            if previous[key]!=contract[key]:raise ValueError('Resume contract mismatch: '+key)
        for key in ('mode','seed','lr','sweeps','train_limit','eval_limit'):
            if previous['arguments'][key]!=vars(args)[key]:raise ValueError('Resume argument mismatch: '+key)
        if (out/('epoch_%03d'%(start_epoch+1))).exists():
            raise ValueError('Preserve interrupted epoch directory under a different name before resuming')
    write(out/'contract.json',contract)
    write(out/'status.json',dict(status='RUNNING',stage='initializing',epoch=start_epoch))
    train=MultiResolutionStream(manifest['train'],configs,args.sweeps,args.seed,True)
    val=MultiResolutionStream(manifest['val'],configs,args.sweeps,args.seed)
    criterion=PointPillarDirLoss(base['loss']['args']).cuda()
    if args.eval_only:
        d=out/'eval';d.mkdir();evaluate(model,val,args,d);write(out/'status.json',dict(status='COMPLETED',stage='eval_only'));return
    for epoch in range(start_epoch+1,args.epochs+1):
        model.train();losses=[];tick=time.monotonic();steps=0
        for sample in data_loader(train,args.workers,True,args.seed+epoch):
            if args.train_limit and steps>=args.train_limit:break
            optimizer.zero_grad(set_to_none=True);total=0.
            for k,c in enumerate(configs):
                select_resolution(model,c,k);ego=to_device(sample['egos'][k],torch.device('cuda'))
                prediction=model(ego);loss=criterion(prediction,ego['label_dict'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();total+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            optimizer.step();steps+=1;losses.append(total)
            if steps%50==0:
                write(out/'status.json',dict(status='RUNNING',stage='train',epoch=epoch,steps=steps,loss=total,seconds=time.monotonic()-tick))
                print('TRAIN',args.mode,epoch,steps,total,flush=True)
        scheduler.step();epoch_dir=out/('epoch_%03d'%epoch);epoch_dir.mkdir()
        model.eval();write(out/'status.json',dict(status='RUNNING',stage='validation',epoch=epoch,steps=steps))
        evaluation=evaluate(model,val,args,epoch_dir)
        history.append(dict(epoch=epoch,steps=steps,loss=float(np.mean(losses)),evaluation=evaluation))
        saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,history=history,
                   rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        torch.save(saved,str(out/'latest.tmp'));os.replace(str(out/'latest.tmp'),str(out/'latest.pth'))
        if evaluation['ap_valid'] and evaluation['results'][0]['ap70']>=max(h['evaluation']['results'][0]['ap70'] for h in history if h['evaluation']['ap_valid']):
            torch.save(dict(model=model.state_dict(),epoch=epoch,contract=contract),str(out/'best.tmp'));os.replace(str(out/'best.tmp'),str(out/'best.pth'))
        write(out/'history.json',history)
    write(out/'status.json',dict(status='COMPLETED',epochs=args.epochs,stop_reason='epoch_budget_not_convergence'))


if __name__=='__main__':main()
