"""Paired immutable-anchor repair, with/without current-E spatial readout."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_anchor_residual import (
    AnchorResidualModules, AnchorResidualDetector, refresh_reason, predict)
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_light_repair import foreground_head_loss
from opencood.models.sub_modules.int_local_state import feature_distillation
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def original_frame(model, ego, state, meta):
    capture = {}
    hooks = [
        model.backbone.resnet.layer0.register_forward_hook(
            lambda m,x,y:capture.update(e=y.detach())),
        model.cls_head.register_forward_pre_hook(
            lambda m,x:capture.update(h=x[0].detach()))]
    try:
        with torch.no_grad():
            pred, state, info = model.step(ego, state, meta)
    finally:
        for hook in hooks:
            hook.remove()
    return pred, capture['e'], capture['h'], state, info


def training_features(modules, evidence, target, cache, meta, reset, bounds):
    reason = refresh_reason(cache, meta, reset, 3)
    if reason:
        cache = dict(value=target, pose=meta['pose'],
                     stamp=int(meta['timestamp_us']), frames=0)
        value = target
        valid = torch.ones_like(target[:,:1], dtype=torch.bool)
    else:
        old, valid = warp_bev(cache['value'], cache['pose'], meta['pose'], bounds)
        value = modules.repair(torch.cat([evidence, old, valid.to(old.dtype)], 1))
        cache = dict(cache, frames=cache['frames']+1)
    return modules.decode(evidence, value, valid), cache, reason


class ConvCounter:
    """Executed standard conv MACs; timing is always measured without hooks."""
    def __init__(self, model, modules):
        self.total = 0
        self.hooks = []
        def count(layer, inputs, output):
            kh,kw = layer.kernel_size
            if isinstance(layer, nn.ConvTranspose2d):
                n,c,h,w = inputs[0].shape
                self.total += n*h*w*c*(layer.out_channels//layer.groups)*kh*kw
            else:
                self.total += output.numel()*(layer.in_channels//layer.groups)*kh*kw
        for root in (model,modules):
            for layer in root.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    self.hooks.append(layer.register_forward_hook(count))

    def close(self):
        for hook in self.hooks:
            hook.remove()


def engineering(model, cfg, rows, modules):
    initial = copy.deepcopy(modules.state_dict())
    full = AnchorResidualDetector(model,cfg,modules,1)
    cache_detector = AnchorResidualDetector(model,cfg,modules,3)
    states = [None]*4
    train_cache = None
    errors, depths, costs = [], [], {}
    count = [0,0,0]
    heavy = [model.backbone.resnet.layer1,model.backbone.resnet.layer2,model.backbone.deblocks[0]]
    def hit(i):
        def callback(m,x,y):
            count[i] += 1
        return callback
    hooks = [m.register_forward_hook(hit(i)) for i,m in enumerate(heavy)]
    counter = ConvCounter(model,modules)
    _,stream = loader(rows[:32],cfg,0,303)
    try:
        for index,sample in enumerate(stream):
            ego = to_device(sample['ego'],torch.device('cuda')); meta=sample['meta']
            before = counter.total
            target,e,h,states[0],info = original_frame(model,ego,states[0],meta)
            original_cost = counter.total-before
            with torch.no_grad():
                before = counter.total
                actual,states[1],_ = full.step(ego,states[1],meta)
                full_cost = counter.total-before
                expected = predict(model,modules.decode(e,h,torch.ones_like(h[:,:1],dtype=torch.bool)))
                assert all(torch.allclose(actual[k],expected[k],atol=1e-5,rtol=1e-5) for k in actual)
                if index == 0:
                    assert all(torch.equal(actual[k],target[k]) for k in actual)
                    # Exercise non-identity corrections and readout in the
                    # deployment/training parity check, then restore weights.
                    for name,module in modules.named_modules():
                        if name.endswith('net'):
                            torch.nn.init.normal_(module[-1].weight,std=.001)
                before = counter.total; calls=list(count)
                actual,states[2],cached_info = cache_detector.step(ego,states[2],meta)
                cache_cost = counter.total-before
                partial = cached_info['anchor_partial']
                assert [x-y for x,y in zip(count,calls)] == ([0,0,0] if partial else [1,1,1])
                key = 'partial' if partial else 'full'
                if key in costs:
                    assert costs[key] == cache_cost
                costs[key] = cache_cost
                costs['original'] = original_cost
                assert full_cost == costs.get('full',full_cost)
            old_cache = train_cache
            features,train_cache,reason = training_features(
                modules,e,h,train_cache,meta,info['reset'],cfg['model']['args']['lidar_range'])
            expected = predict(model,features)
            error = max(float((expected[k]-actual[k]).abs().max()) for k in actual)
            errors.append(error)
            assert all(torch.allclose(actual[k],expected[k],atol=1e-4,rtol=1e-4) for k in actual)
            assert partial == (reason is None)
            if partial:
                depths.append(train_cache['frames'])
                assert train_cache['value'] is old_cache['value']
                assert train_cache['stamp'] == old_cache['stamp']
                assert states[2]['cache']['stamp'] == train_cache['stamp']
                loss = foreground_head_loss(expected,target)+feature_distillation(features,h,target)
                modules.zero_grad(set_to_none=True); loss.backward()
                assert all(p.grad is None for p in model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in modules.parameters())
        assert set(depths)=={1,2}
        optimizer = torch.optim.AdamW(modules.parameters(),lr=1e-3)
        optimizer.step()
        assert any(not torch.equal(v,initial[k]) for k,v in modules.state_dict().items())
    finally:
        for hook in hooks:
            hook.remove()
        counter.close()
        modules.load_state_dict(initial,strict=True)
        modules.zero_grad(set_to_none=True)
    return dict(status='PASSED',frames=len(errors),prediction_max_error=max(errors),
                partial_depths=sorted(set(depths)),heavy_calls_on_partial=0,
                immutable_anchor_checked=True,discarded_probe_weights=True,
                conv_macs_per_frame=costs)


def evaluate(model, cfg, rows, modules, out, costs, epoch):
    modules.eval()
    modes = dict(original=model,
                 full=AnchorResidualDetector(model,cfg,modules,1),
                 cached=AnchorResidualDetector(model,cfg,modules,3))
    states={k:None for k in modes}; times={k:[] for k in modes}
    aps={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for k in modes}
    partial=0; original_weights={k:v.detach().clone() for k,v in modules.state_dict().items()}
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda')); meta=sample['meta']
            names=list(modes); names=names[index%3:]+names[:index%3]
            for name in names:
                torch.cuda.synchronize(); tick=time.perf_counter()
                pred,states[name],info=modes[name].step(ego,states[name],meta)
                torch.cuda.synchronize(); ms=1000*(time.perf_counter()-tick)
                assert all(torch.isfinite(v).all() for v in pred.values())
                if index>=10:times[name].append(ms)
                if name=='cached':partial+=int(info['anchor_partial'])
                log.write(json.dumps(dict(frame=meta['frame'],mode=name,ms=ms,
                                          partial=info.get('anchor_partial',False)))+'\n')
                if meta['supervised']:
                    data=dict(ego=ego)
                    boxes,scores=dataset.post.post_process(data,dict(ego=pred))
                    gt=dataset.post.generate_gt_bbx(data)
                    for t in aps[name]:
                        eval_utils.caluclate_tp_fp(boxes,scores,gt,aps[name],t)
            if (index+1)%200==0:
                write_json(out.parent/'status.json',dict(status='RUNNING',phase='validation',
                    epoch=epoch,scans=index+1,total=len(rows)))
    assert index+1==len(rows)
    assert all(torch.equal(v,original_weights[k]) for k,v in modules.state_dict().items())
    result={}
    for name in modes:
        macs=costs['original'] if name=='original' else costs['full']
        if name=='cached':
            macs=(costs['partial']*partial+costs['full']*(len(rows)-partial))/len(rows)
        result[name]=dict(ap70=eval_utils.calculate_ap(aps[name],.7)[0],
            ap50=eval_utils.calculate_ap(aps[name],.5)[0],mean_ms=float(np.mean(times[name])),
            p95_ms=float(np.percentile(times[name],95)),conv_gflops=2*macs/1e9,
            conv_reduction_vs_original=1-macs/costs['original'])
    assert abs(result['original']['ap70']-.758736104616861)<1e-8
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),
                partial_frames=partial,deep_call_fraction=1-partial/len(rows),results=result,
                conv_scope='Conv2d+ConvTranspose2d standard MACs x2; excludes VFE Linear, activations, warps, BN and other non-conv operations. Transposed conv counts input positions including boundary products. Not complete model FLOPs or hardware instructions.',
                timing_scope='Same GPU rotating modes; synchronized model, excludes IO/H2D/postprocess; training concurrent on other GPUs may contend for host resources.',
                repair_area='Whole H on every partial frame; no selective spatial skipping yet.')
    write_json(out/'report.json',report)
    return report


def worker(a):
    out=Path(a.output)
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,
                    periodic_last=0,max_age=.25).items():setattr(a,k,v)
    model,cfg,val,provenance=setup(a)
    assert provenance['checkpoint']['sha256']=='c7bbee9b4fdefc031a61b240d6c504665d66c0c788610bb36c97d34118f8a6b8'
    for p in model.parameters():p.requires_grad_(False)
    frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
    source_args=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(source_args['manifest']).read_text())['train']
    torch.manual_seed(303); modules=AnchorResidualModules(a.arm=='fusion').cuda()
    contract=dict(arm=a.arm,seed=303,refresh_period=3,max_age=.25,anchor_writeback=False,
        evaluation_only=a.diagnostic,
        epochs_max=a.epochs,minimum_epochs=6,plateau_patience=4,minimum_ap_gain=.001,
        original_sha256=provenance['checkpoint']['sha256'],data_contract=provenance['contract'],
        train_scans=len(train),val_scans=len(val),trainable_parameters=sum(p.numel() for p in modules.parameters()),
        frozen='Original INT GRU, Pillar, shallow/deep stages, decoder, shrink and heads; fixed BN',
        loss='Partial frames only: supervised GT where available + 0.2 foreground head KD + 0.5 normalized feature KD on final readout; same formula both arms',
        initialization='Zero last residual convolutions; identical repair initialization across arms',
        optimizer='AdamW lr1e-3 wd1e-4 clip5; ReduceLROnPlateau cached AP, factor0.5 patience1 min_lr1e-5',
        protocol='Synchronous causal existing DAIR sequences; no augmentation/artificial delay; immutable full-refresh H. Full network for training targets only; no deep/decoder calls on partial deployment frames.',
        interpretations='Architecture+capacity comparison, not parameter-matched causal proof; three modes separate full precision from incremental cache loss.',
        next_step='First validate dense small repair/readout; only then deploy region-selective repair.')
    write_json(out/'training_contract.json',contract);write_json(out/'provenance.json',provenance)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    assert tests.returncode==0
    gate=engineering(model,cfg,train,modules);write_json(out/'engineering.json',gate)
    if a.diagnostic:
        directory=out/'epoch_000';directory.mkdir()
        report=evaluate(model,cfg,val,modules,directory,gate['conv_macs_per_frame'],0)
        report.update(diagnostic='Zero residual, immutable warped anchor; untrained modules still execute. Accuracy control, not optimized warp-only latency.',
                      optimizer_steps=0)
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(status='COMPLETED',diagnostic=True))
        return
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    optimizer=torch.optim.AdamW(modules.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,
        patience=1,threshold=.001,threshold_mode='abs',min_lr=1e-5)
    best=significant=-1.; stale=0; history=[]
    for epoch in range(1,a.epochs+1):
        modules.train(); state=cache=None; losses=[];depths={1:0,2:0};tick=time.monotonic()
        directory=out/('epoch_%03d'%epoch);directory.mkdir()
        _,stream=loader(train,cfg,4,303+epoch)
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            assert meta['frame']==train[index]['frame']
            target,e,h,state,info=original_frame(model,ego,state,meta)
            features,cache,reason=training_features(modules,e,h,cache,meta,info['reset'],
                                                    cfg['model']['args']['lidar_range'])
            if reason is None:
                pred=predict(model,features)
                loss=.2*foreground_head_loss(pred,target)+.5*feature_distillation(features,h,target)
                if meta['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(modules.parameters(),5.)
                assert torch.isfinite(norm)
                optimizer.step();losses.append(float(loss.detach()));depths[cache['frames']]+=1
            if (index+1)%200==0:
                write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,
                    scans=index+1,total=len(train),loss=float(np.mean(losses)) if losses else None))
                print('TRAIN',epoch,index+1,'loss',np.mean(losses) if losses else None,flush=True)
        assert index+1==len(train) and min(depths.values())>0
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
        modules.eval()
        path=directory/'modules.pth'
        torch.save(dict(modules=modules.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(model,cfg,val,modules,directory,gate['conv_macs_per_frame'],epoch)
        ap=report['results']['cached']['ap70'];scheduler.step(ap)
        if ap>best:
            best=ap;write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(path),sha256=digest(path)))
        if ap>significant+.001:significant=ap;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),updates_by_depth=depths,
                            validation=report,seconds=time.monotonic()-tick,lr=optimizer.param_groups[0]['lr']))
        write_json(out/'history.json',history)
        torch.save(dict(modules=modules.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch,history=history,contract=contract),
            str(out/'latest.pth'))
        if epoch>=6 and stale>=4 and optimizer.param_groups[0]['lr']<=.0001:break
    write_json(out/'report.json',dict(status='COMPLETED',epochs=len(history),best=json.loads((out/'best.json').read_text()),
        stop_reason='plateau' if epoch<a.epochs else 'budget_not_convergence'))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=len(history)))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--arm',choices=['direct','fusion'],required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--diagnostic',action='store_true',help='Full zero-residual baseline; no training')
    a=p.parse_args();out=Path(a.output)
    if not 1<=a.epochs<=12:p.error('Maximum initial budget is 12 epochs')
    if a.worker:
        def expired(signum,frame):raise TimeoutError('Anchor experiment exceeded 8 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(28800)
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_anchor_residual']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                              stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),
        timeout_seconds=28800,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':
    main()
