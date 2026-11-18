"""Matched fixed-mask versus whole-map training of the ordinary deep gate."""
import argparse
import json
from pathlib import Path
import random
from statistics import mean
import subprocess

import numpy as np
import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_block_adapt import (
    DenseOrdinaryReference, TRAINED, WINDOW, capture_rng, equal_tree, fixed_clip,
    make_trainable, restore_rng, tensor_digest,
)
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import CAP, SEED, TILE, checked_plan, clip_streams
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.run_int_epochs import atomic_save
from opencood.utils import eval_utils


TRAIN_ARMS = ('whole_trained', 'local_trained')
EVAL_ARMS = ('p5', 'frozen_local', *TRAIN_ARMS)
EVAL_EPOCHS = (0, 1, 4, 8)
EPOCHS = EVAL_EPOCHS[-1]
GRAD_CLIP = 5.


def pack_mask(mask):
    return None if mask is None else mask.flatten().nonzero().flatten().tolist()


def compile_trace(source, cfg, rows, plan):
    cache = DeepBlockCache(source, TILE, 'light', crop_first=True, planner=plan_host_coalesced_refresh)
    clips = []
    for _, ids, _, stream in clip_streams(plan, rows, cfg):
        scans = fixed_clip(cache, ids, stream)
        result = []
        for scan in scans:
            row = dict(frame=scan.frame, refresh=scan.refresh, supervised=scan.target is not None)
            if scan.refresh is None:
                row.update(shape=list(scan.valid.shape), optional=pack_mask(scan.request),
                           written=pack_mask(scan.written), valid=pack_mask(scan.valid))
            result.append(row)
        clips.append(result)
    return clips


def warm_batches(trace):
    start, count = 0, 0
    for index, row in enumerate(trace):
        if row['refresh'] is not None:
            if count:
                yield start, index
            yield index, index+1
            start, count = index+1, 0
        else:
            if count == 0:
                start = index
            count += 1
            if count == WINDOW:
                yield start, index+1
                count = 0
    if count:
        yield start, len(trace)


def trace_masks(row, device):
    if row['refresh'] is not None:
        return None
    masks = {}
    for name in ('optional', 'written', 'valid'):
        indices = row[name]
        mask = None
        if indices is not None:
            mask = torch.zeros(row['shape'], device=device, dtype=torch.bool)
            mask.view(-1)[indices] = True
        masks[name] = mask
    return masks


def load_clip(ids, stream, trace, source):
    entries = []
    for index, sample in enumerate(stream):
        row = trace[index]
        meta = sample['meta']
        assert row['frame'] == meta['frame'] == ids[index]
        assert row['supervised'] == bool(meta['supervised'])
        ego = to_device(sample['ego'], torch.device('cuda'))
        spatial = spatial_input(source, ego)
        entries.append((meta, ego, spatial, trace_masks(row, spatial.device)))
    assert len(entries) == len(ids) == len(trace)
    return entries


def local_request(masks):
    if masks is None:
        return None

    def check(frame, mandatory):
        optional = masks['optional']
        written = mandatory if optional is None else mandatory | optional
        assert torch.equal(frame.valid, masks['valid']) and torch.equal(written, masks['written'])
        return optional

    return check


def all_write(frame, mandatory):
    return torch.ones_like(mandatory)


def train_clip(name, model, opt, trace, entries, criterion, raw):
    state = None
    reference = DenseOrdinaryReference(model)
    records = []
    for start,end in warm_batches(trace):
        chunk = trace[start:end]
        labels = [row['frame'] for row in chunk if row['supervised'] and row['refresh'] is None]
        losses = []
        opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(bool(labels)):
            for meta, ego, spatial, masks in entries[start:end]:
                request = all_write if name == 'whole_trained' and masks is not None else local_request(masks)
                pred, state, info = reference.step_spatial(spatial,state,meta,request)
                assert info['full_refresh'] == trace[start]['refresh']
                if meta['frame'] in labels:
                    losses.append(criterion(pred, ego['label_dict']))
            if labels:
                loss = torch.stack(losses).mean()
                assert bool(torch.isfinite(loss))
                loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None for p in parameters)
                norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
                assert bool(torch.isfinite(norm))
                opt.step()
                rec = dict(frames=[r['frame'] for r in chunk], labels=labels,
                           loss=float(loss.detach()), grad_norm=float(norm))
                records.append(rec)
                raw.write(json.dumps(dict(arm=name, **rec),allow_nan=False)+'\n')
        state = dict(state, deep=state['deep'].detach())
    return records


def train_epoch(epoch, parent, models, optimizers, cfg, rows, plan, traces, out, frozen_digest):
    directory = out / ('epoch_%02d' % epoch)
    directory.mkdir()
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    by_arm = {name:[] for name in TRAIN_ARMS}
    with (directory/'updates.jsonl').open('x',buffering=1) as raw:
        for clip, ids, _, stream in clip_streams(plan,rows,cfg):
            with torch.no_grad():
                entries = load_clip(ids,stream,traces[clip],parent)
            for name in TRAIN_ARMS:
                by_arm[name].extend(train_clip(name, models[name], optimizers[name], traces[clip], entries, criterion, raw))
            del entries
            write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,clips=clip+1,
                       updates={k:len(v) for k,v in by_arm.items()}))
    assert all(tensor_digest(model.state_dict(), TRAINED)==frozen_digest for model in models.values())
    ids = [[(r['frames'],r['labels']) for r in by_arm[k]] for k in TRAIN_ARMS]
    assert ids[0] and ids[0] == ids[1]
    return {k:dict(updates=len(v),labels=sum(len(r['labels']) for r in v),
                   mean_loss=mean(r['loss'] for r in v), mean_grad_norm=mean(r['grad_norm'] for r in v))
            for k,v in by_arm.items()}


@torch.no_grad()
def evaluate(epoch, parent, models, cfg, rows, plan, traces, out, include_scenes=False):
    directory = out / ('eval_%02d' % epoch)
    directory.mkdir()
    built = {'p5': parent}
    for k,model in {'frozen_local':parent, **models}.items():
        built[k] = DeepBlockCache(model,TILE,'light',crop_first=True,planner=plan_host_coalesced_refresh)
    stats = {k:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for k in EVAL_ARMS}
    totals = dict.fromkeys(EVAL_ARMS,0)
    scans, annotated = 0, 0
    scene_slices = []
    with (directory/'frames.jsonl').open('x',buffering=1) as raw:
        for clip, ids, dataset, stream in clip_streams(plan,rows,cfg):
            start = {name:{t:(len(record['score']),record['gt']) for t,record in arm.items()}
                     for name,arm in stats.items()} if include_scenes else None
            states = dict.fromkeys(EVAL_ARMS)
            entries = load_clip(ids,stream,traces[clip],parent)
            for meta, ego, spatial, masks in entries:
                batch = dict(ego=ego)
                gt = dataset.post.generate_gt_bbx(batch) if meta['supervised'] else None
                annotated += int(meta['supervised'])
                gold, cost = None, None
                count = {}
                for name in EVAL_ARMS:
                    model = built[name]
                    req = None if name == 'p5' else local_request(masks)
                    (pred,states[name],info),flops = measured_step(model,spatial,states[name],meta,req)
                    assert all(bool(torch.isfinite(t).all()) for t in pred.values())
                    assert info['full_refresh'] == traces[clip][ids.index(meta['frame'])]['refresh']
                    if name == 'frozen_local':
                        gold, cost = pred, flops
                    elif name in TRAIN_ARMS:
                        assert cost == flops
                        if epoch == 0:
                            assert all(torch.equal(t,gold[k]) for k,t in pred.items())
                    totals[name] += flops
                    count[name] = flops
                    if gt is not None:
                        boxes,scores = dataset.post.post_process(batch,dict(ego=pred))
                        for t in (.5,.7):
                            eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
                scans += 1
                raw.write(json.dumps(dict(frame=meta['frame'],clip=clip,supervised=gt is not None,
                    core_conv_flops=count))+'\n')
            del entries
            if include_scenes:
                scene_slices.append(dict(scene=plan['scenes'][clip][0], clip=clip,
                    arms={name:{str(t):[start[name][t][0],len(record['score']),record['gt']-start[name][t][1]]
                                for t,record in arm.items()} for name,arm in stats.items()}))
            write_json(out/'status.json',dict(status='RUNNING',phase='eval',epoch=epoch,scans=scans))
    assert scans == sum(map(len,plan['clips'])) and annotated == plan['labelled']
    write_json(directory/'ap_records.json',stats)
    if include_scenes:
        write_json(directory/'scene_slices.json',scene_slices)
    write_json(directory/'sort_orders.json',dict(source_sha256=digest(directory/'ap_records.json'),
        orders={key:{str(t):np.argsort(-np.asarray(row['score'])).tolist() for t,row in values.items()}
                for key,values in stats.items()}))
    metrics = {k:dict(ap50=eval_utils.calculate_ap(stats[k],.5)[0],ap70=eval_utils.calculate_ap(stats[k],.7)[0],
                     core_conv_flops_mean=totals[k]/scans,model_step_ms_mean=None)
               for k in EVAL_ARMS}
    report = dict(scans=scans, supervised=annotated, gt=stats['p5'][.5]['gt'], metrics=metrics)
    write_json(directory/'report.json',report)
    return report


def identity(args):
    return dict(plans={'train':digest(Path(args.train_plan)),'val':digest(Path(args.val_plan))},
        source_sha256=SOURCE_SHA, code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        seed=SEED, epochs=EPOCHS, eval_epochs=EVAL_EPOCHS, window=WINDOW, grad_clip=GRAD_CLIP, tile=TILE, cap=CAP,
        period=5,max_age=.55,trainable=TRAINED,optimizer='AdamW(lr=1e-4,weight_decay=1e-4)',
        supervision='Mean PointPillarDirLoss of labeled ordinary frames per fixed nonoverlapping chunk of up to three. Refresh full with gradients detached.',
        evaluation='Identical fixed residual write requests for the three local arms; P5 kept whole map. No latency in quality run.')


def checkpoint(models,optimizers,history,contract,epoch):
    return dict(identity=contract, epoch=epoch, history=history, rng=capture_rng(),
        weights={k:{n:m.state_dict()[n].detach().cpu() for n in TRAINED} for k,m in models.items()},
        optimizers={k:o.state_dict() for k,o in optimizers.items()})


def restore(saved,models,optimizers,contract):
    assert saved['identity'] == contract
    for k,m in models.items():
        opt = optimizers[k]
        with torch.no_grad():
            for name,p in m.named_parameters():
                if name in TRAINED:
                    p.copy_(saved['weights'][k][name])
                    assert torch.equal(p.detach().cpu(),saved['weights'][k][name])
        opt.load_state_dict(saved['optimizers'][k])
        opt.zero_grad(set_to_none=True)
        assert equal_tree(opt.state_dict(),saved['optimizers'][k])
    restore_rng(saved['rng'])
    assert equal_tree(capture_rng(),saved['rng'])


def run(args,out):
    tr,va = checked_plan(args.train_plan),checked_plan(args.val_plan)
    assert tr['split']=='train' and va['split']=='val'
    assert {s[0] for s in tr['scenes']}.isdisjoint(s[0] for s in va['scenes'])
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    args.kind = 'quality'
    source,cfg,train_rows,_ = build(args)
    args.kind = 'geometry'
    val_source,vcfg,val_rows,_ = build(args)
    original = model_digest(source)
    assert model_digest(val_source)==original
    del val_source
    source.period,source.max_age = 5,.55
    frozen = tensor_digest(source.state_dict(),TRAINED)
    models,optimizers = {},{}
    for name in TRAIN_ARMS:
        models[name],optimizers[name] = make_trainable(source)
    spec = identity(args)
    spec['frozen_digest'] = frozen
    trace_file = out/'fixed_traces.json'
    if args.resume:
        trace_file.write_bytes(Path(args.trace).read_bytes())
    else:
        with torch.no_grad():
            write_json(trace_file,dict(train=compile_trace(source,cfg,train_rows,tr),
                                       val=compile_trace(source,vcfg,val_rows,va)))
    spec['trace_sha256'] = digest(trace_file)
    traces = json.loads(trace_file.read_text())
    write_json(out/'contract.json',dict(identity=spec,device=torch.cuda.get_device_name(),torch=torch.__version__,
                                      stop_after=args.stop_after,resume=args.resume))
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu')
        restore(saved,models,optimizers,spec)
        start,history = saved['epoch']+1,saved['history']
    else:
        start = 1
        history = [dict(epoch=0,eval=evaluate(0,source,models,vcfg,val_rows,va,traces['val'],out))]
    assert 1 <= start <= args.stop_after <= EPOCHS
    for epoch in range(start,args.stop_after+1):
        train = train_epoch(epoch,source,models,optimizers,cfg,train_rows,tr,traces['train'],out,frozen)
        val = evaluate(epoch,source,models,vcfg,val_rows,va,traces['val'],out) if epoch in EVAL_EPOCHS else None
        assert model_digest(source)==original
        history.append(dict(epoch=epoch,train=train,eval=val))
        saved = out/('after_epoch_%02d.pth' % epoch)
        state = checkpoint(models,optimizers,history,spec,epoch)
        atomic_save(state,saved)
        reopened = torch.load(saved,map_location='cpu')
        assert equal_tree(reopened,state), 'Atomic checkpoint roundtrip differs'
        write_json(out/'latest_checkpoint.json',dict(epoch=epoch,path=str(saved),sha256=digest(saved)))
        write_json(out/'history.json',history)
    write_json(out/'report.json',dict(status='COMPLETED',epoch=args.stop_after,full_budget=args.stop_after==EPOCHS,
        parameters=sum(p.numel() for p in source.parameters()),trainable_parameters=sum(p.numel() for p in models[TRAIN_ARMS[0]].parameters() if p.requires_grad),
        last_eval=next(row['eval'] for row in reversed(history) if row['eval'] is not None),history=history))
    write_json(out/'status.json',dict(status='COMPLETED',epoch=args.stop_after,full_budget=args.stop_after==EPOCHS))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('train-plan','val-plan','output','root','checkpoint','trained','readout'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--resume')
    p.add_argument('--trace')
    p.add_argument('--stop-after',required=True,type=int)
    args=p.parse_args()
    if bool(args.resume)!=bool(args.trace) or not 1<=args.stop_after<=EPOCHS:
        p.error('Resume requires its fixed trace, and stop-after must be in 1..8')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs_and_trace'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
