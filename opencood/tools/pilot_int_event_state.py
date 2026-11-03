"""Fixed detection/unit-event pilot. Real-model gates use the sole quality GPU.

Preflight and optimization are separate, locally verified runs. This tool does
not score AP, choose checkpoints, change quanta or claim sparse training cost.
"""
from opencood.tools.int_joint_adaptation_setup import *
from opencood.models.sub_modules.int_readout_event_objective import EventObjectiveReadout
from opencood.models.sub_modules.int_event_training_loss import (
    window_objective, gradient_summary, READOUT_DENSE_MAC, UNIT_ENERGY_RATIO, EVENT_WEIGHT)
from opencood.tools.pilot_int_joint_state_qat import frozen_digest, buffer_digest
from opencood.tools.pilot_int_gated_state import detach_state
from opencood.tools.check_int_fine_readout_restore import tree_digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def make_model(source, cal, rc, observed=True):
    model = JointStateQATDetector(source, cal['layers'], rc['quanta'], True).cuda().eval()
    if observed:
        model.core.readout = EventObjectiveReadout(source.core.readout, rc['quanta']).cuda().eval()
    model.open_ordinary_updates()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2449664
    return model


def initial_pairs(source, cal, rc, cfg, training, original, out):
    # This gate uses the complete first eligible window, without any optimizer.
    length = original['expected_update_indices'][0] + 1
    models = {n:make_model(source, cal, rc, n != 'plain') for n in ('plain','control','event')}
    states = {n:None for n in models}; before = {n:model_digest(m) for n,m in models.items()}
    assert len(set(before.values())) == 1
    criterion = PointPillarDirLoss(cfg['loss']['args']).cuda()
    _, stream = loader(training[:length], cfg, 2, 303)
    with torch.no_grad(), (out/'initial_pairs.jsonl').open('x', buffering=1) as log:
        for i,sample in enumerate(stream):
            ego = to_device(sample['ego'], torch.device('cuda')); meta = sample['meta']; records = {}
            for name,model in models.items():
                if name != 'plain':model.core.readout.begin_frame(bool(original['expected_refresh_reasons'][i]))
                pred,states[name],info = model.step(ego, states[name], meta)
                assert info['full_refresh'] == original['expected_refresh_reasons'][i]
                records[name] = dict(prediction_hash=tree_digest(pred), task_hash=tree_digest(states[name]),
                                     loss=float(criterion(pred,ego['label_dict'])) if meta['supervised'] else None)
            assert records['plain'] == records['control'] == records['event']
            log.write(json.dumps(dict(index=i,frame=meta['frame'],arms=records))+'\n')
    assert all(model_digest(m) == before[n] for n,m in models.items())
    del models,states;torch.cuda.empty_cache()
    return dict(scans=length,plain_control_event_exact=True,weights_unchanged=True)


def component_preflight(model, cfg, training, original, out, name):
    before=model_digest(model);frozen=frozen_digest(model);buffers=buffer_digest(model)
    names,opened=zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();state=None;losses=[];units=[]
    ends=set(original['window_end_indices']);eligible=set(original['expected_update_indices'])
    torch.cuda.reset_peak_memory_stats();_,stream=loader(training,cfg,2,303)
    with (out/(name+'_preflight_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];readout=model.core.readout
            readout.begin_frame(bool(original['expected_refresh_reasons'][i]))
            pred,state,info=model.step(ego,state,meta,detach_state=False)
            assert info['full_refresh']==original['expected_refresh_reasons'][i]
            assert all(bool(torch.isfinite(x).all()) for x in pred.values())
            if meta['supervised']:losses.append(criterion(pred,ego['label_dict']))
            units.append(readout.event_units)
            log.write(json.dumps(dict(index=i,frame=meta['frame'],refresh=info['full_refresh'],
                                      weighted_mac=int(readout.event_weighted.detach()),unit_ac=int(readout.event_units.detach())))+'\n')
            if i in ends:
                if i in eligible:
                    total,det,event=window_objective(losses,units,name=='event')
                    dg=torch.autograd.grad(det,opened,retain_graph=True,allow_unused=True)
                    eg=torch.autograd.grad(event,opened,retain_graph=True,allow_unused=True)
                    tg=torch.autograd.grad(total,opened,allow_unused=True)
                    components={n:gradient_summary(names,g) for n,g in [('detection',dg),('event',eg),('total',tg)]}
                    assert all(s['finite'] and s['nonzero_tensors']>0 for s in components.values()),components
                    assert all(g is not None for g in tg)
                    assert all(p.grad is None for p in model.parameters())
                    dot=sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(dg,eg) if a is not None and b is not None)
                    cosine=dot/(components['detection']['norm']*components['event']['norm'])
                    assert model_digest(model)==before and frozen_digest(model)==frozen and buffer_digest(model)==buffers
                    result=dict(status='PASSED',scans=i+1,frame=meta['frame'],components=components,
                                detection_loss=float(det.detach()),event_energy_ratio=float(event.detach()),
                                total_loss=float(total.detach()),gradient_cosine=cosine,
                                weighted_event_to_detection_norm_ratio=EVENT_WEIGHT*components['event']['norm']/components['detection']['norm'],
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                objective_reference_bytes=readout.reference_bytes(),task_state_bytes=info['cache_bytes'],
                                weights_unchanged=True,optimizer_steps=0,parameters=sum(p.numel() for p in model.parameters()),
                                trainable_parameters=2449664,flops=None,model_step_ms_mean=None,ap50=None,ap70=None)
                    write_json(out/(name+'_preflight.json'),result);return result
                losses=[];units=[];state=detach_state(state)
    raise RuntimeError('No eligible original update window')


def train_one(model,cfg,training,original,contract,out,name,gold,old_final):
    before=model_digest(model);frozen=frozen_digest(model);buffers=buffer_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();state=None;losses=[];units=[];updates=[];labels=0
    ends=set(original['window_end_indices']);eligible=set(original['expected_update_indices']);ordinary=False
    _,stream=loader(training,cfg,2,303);torch.cuda.reset_peak_memory_stats()
    with (out/(name+'_train_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];readout=model.core.readout
            assert meta['frame']==training[i]['frame']==gold[i]['frame']
            readout.begin_frame(bool(original['expected_refresh_reasons'][i]))
            pred,state,info=model.step(ego,state,meta,detach_state=False)
            assert all(bool(torch.isfinite(x).all()) for x in pred.values())
            assert info['full_refresh']==original['expected_refresh_reasons'][i];ordinary|=not bool(info['full_refresh'])
            row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],refresh=info['full_refresh'],optimizer_step=False)
            extra=dict(index=i,weighted_event_mac=int(readout.event_weighted.detach()),unit_spike_ac=int(readout.event_units.detach()),
                       objective_reference_bytes=readout.reference_bytes(),task_state_bytes=info['cache_bytes'])
            units.append(readout.event_units)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if i in eligible:
                    assert losses and ordinary
                    objective,det,event=window_objective(losses,units,name=='event');assert bool(torch.isfinite(objective))
                    objective.backward()
                    for n,p in model.named_parameters():
                        if p.requires_grad:assert p.grad is not None and bool(torch.isfinite(p.grad).all()),n
                        else:assert p.grad is None,n
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()),optimizer_step=True)
                    extra.update(window_detection_loss=float(det.detach()),window_event_energy_ratio=float(event.detach()),window_scans=len(units),window_labels=len(losses))
                    optimizer.step();updates.append(meta['frame']);assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];units=[];ordinary=False;state=detach_state(state)
                if row['optimizer_step'] and len(updates)%64==0:
                    torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                                    optimizer=optimizer.state_dict(),own_state=detach_state(state,cpu=True),
                                    event_previous=[x.detach().cpu() for x in readout.event_previous],next_scan=i+1,updates=updates,
                                    contract=contract,mode=name,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),
                               out/('%s_update_%03d.pth'%(name,len(updates))))
            if name=='control':
                extra['original_trajectory_exact']=row==gold[i]
            log.write(json.dumps(dict(training=row,event=extra))+'\n')
            if name=='control':assert row==gold[i],('Zero-coefficient original trajectory mismatch',i)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',mode=name,scans=i+1,total=2048,updates=len(updates)))
    assert updates==original['expected_update_frames'] and labels==1396 and len(updates)==389
    assert frozen_digest(model)==frozen and buffer_digest(model)==buffers and model_digest(model)!=before
    targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    if name=='control':
        assert tree_digest(targets)==tree_digest(old_final['targets'])
        assert tree_digest(optimizer.state_dict())==tree_digest(old_final['optimizer'])
    path=out/(name+'_final.pth')
    torch.save(dict(targets=targets,optimizer=optimizer.state_dict(),own_state=detach_state(state,cpu=True),
                    event_previous=[x.detach().cpu() for x in model.core.readout.event_previous],next_scan=2048,updates=updates,
                    contract=contract,mode=name,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),path)
    return dict(scans=2048,labels=labels,updates=len(updates),checkpoint_sha256=digest(path),target_hash=tree_digest(targets),
                control_original_trajectory_exact=name=='control',parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=2449664,flops=None,model_step_ms_mean=None,ap50=None,ap70=None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),frozen_weights_bn_unchanged=True)


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare(a,out);del base,exports;torch.cuda.empty_cache()
    full=json.loads(Path(a.full_verification).read_text());resume=json.loads(Path(a.resume_verification).read_text());event=json.loads(Path(a.event_verification).read_text())
    assert full['status']==resume['status']==event['status']=='VERIFIED'
    assert full['run']=='R-223' and resume['run']=='R-230' and event['run']=='R-226'
    assert full['trained_checkpoint_sha256']==resume['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    event_contract=Path(a.event_source)/'contract.json'
    assert digest(event_contract)==event['source_sha256']['contract.json']
    assert json.loads(event_contract.read_text())['trained_checkpoint_sha256']==identity['trained_checkpoint_sha256']
    lp=Path(a.learning_source);original=json.loads((lp/'contract.json').read_text());lv=json.loads(Path(a.learning_verification).read_text())
    goldpath=lp/'qat_train_frames.jsonl';assert digest(goldpath)==lv['source_sha256'][goldpath.name]
    training=rows[768:2816];assert [x['frame'] for x in training]==original['train_frames'] and len(training)==2048
    assert sum(x['supervised'] for x in training)==1396 and len(original['expected_update_frames'])==389
    contract=dict(**identity,original_contract_sha256=digest(lp/'contract.json'),train_frames=original['train_frames'],
                  expected_update_frames=original['expected_update_frames'],window_end_indices=original['window_end_indices'],
                  expected_refresh_reasons=original['expected_refresh_reasons'],expected_update_indices=original['expected_update_indices'],
                  tbptt=4,training_labels=1396,lr=1e-4,weight_decay=1e-4,clip_norm=10.,epochs=1,seed=303,
                  dense_readout_mac_normalizer=READOUT_DENSE_MAC,unit_energy_ratio=UNIT_ENERGY_RATIO,event_weight=EVENT_WEIGHT,
                  full_verification_sha256=digest(Path(a.full_verification)),resume_verification_sha256=digest(Path(a.resume_verification)),
                  event_verification_sha256=digest(Path(a.event_verification)),preflight_only=a.preflight_only,
                  boundary='Same R139+rank128 initialization, frozen quanta/P4/BN. Only intervention: event coefficient0 versus0.1, assumed unitAC/MAC ratio0.1. Label-only detection mean and all-frame unitAC mean within original eligible windows. Detached non-warped sending references; no added task membrane. Dense FP64 training, no AP/deployment FLOPs or latency. One quality GPU. No coefficient search or checkpoint selection.')
    if not a.preflight_only:
        assert a.preflight_verification and a.preflight_source
        v=json.loads(Path(a.preflight_verification).read_text());assert v['status']=='VERIFIED' and v['run']=='R-232'
        pc=Path(a.preflight_source)/'contract.json';assert digest(pc)==v['source_sha256']['contract.json']
        prior=json.loads(pc.read_text());assert all(prior[k]==contract[k] for k in contract if k!='preflight_only')
        contract['preflight_verification_sha256']=digest(Path(a.preflight_verification))
    write_json(out/'contract.json',contract)
    if a.preflight_only:
        initial=initial_pairs(source,cal,rc,cfg,training,original,out);results={}
        for name in ('control','event'):
            model=make_model(source,cal,rc);results[name]=component_preflight(model,cfg,training,original,out,name);del model;torch.cuda.empty_cache()
        assert results['control']['components']['detection']==results['event']['components']['detection']
        assert results['control']['components']['event']==results['event']['components']['event']
        write_json(out/'report.json',dict(status='COMPLETED',initial=initial,preflight=results,trained=False,ap_evaluated=False,timing_valid=False))
    else:
        gold=[json.loads(x) for x in goldpath.read_text().splitlines()];old_final=torch.load(lp/'qat_final.pth',map_location='cpu');results={}
        for name in ('control','event'):
            model=make_model(source,cal,rc);results[name]=train_one(model,cfg,training,original,contract,out,name,gold,old_final);del model;torch.cuda.empty_cache()
        write_json(out/'checkpoint_manifest.json',dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('*.pth'))]))
        write_json(out/'report.json',dict(status='COMPLETED',training=results,trained=True,ap_evaluated=False,timing_valid=False,all_terminals_retained=True))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ARGUMENTS+('full-verification','resume-verification','event-verification','event-source'):p.add_argument('--'+n,required=True)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-verification');p.add_argument('--preflight-source')
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=2700 if a.preflight_only else 14400
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed event pilot budget')));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
