"""Fixed event checkpoint64->128 replay; reserves the sole training/quality GPU.

This is a reproduction experiment, not additional model selection. It must be
launched separately after the active complete quality evaluation has exited.
"""
from opencood.tools.int_event_adaptation_setup import *
from opencood.tools.pilot_int_event_state import make_model
from opencood.models.sub_modules.int_event_training_loss import window_objective
from opencood.tools.pilot_int_joint_state_qat import frozen_digest,buffer_digest
from opencood.tools.pilot_int_gated_state import detach_state
from opencood.tools.check_int_fine_readout_restore import tree_digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def replay(mode,model,cfg,training,source,out,contract,manifest):
    paths={n:source/(mode+'_update_%03d.pth'%n) for n in (64,128)}
    for p in paths.values():assert digest(p)==manifest[p.name]['sha256']
    start=torch.load(paths[64],map_location='cpu');target=torch.load(paths[128],map_location='cpu')
    for count,saved in ((64,start),(128,target)):
        c=dict(saved['contract']);c['trainable_prefixes']=list(c['trainable_prefixes']);assert c==contract and saved['mode']==mode
        assert saved['updates']==contract['expected_update_frames'][:count] and saved['next_scan']==contract['expected_update_indices'][count-1]+1
        assert saved['next_scan']-1 in contract['window_end_indices']
    assert list(start['targets'])==list(target['targets'])
    frozen=frozen_digest(model);buffers=buffer_digest(model);assert load_state_update_targets(model,start['targets'])==2449664
    model.open_ordinary_updates();opened=[p for p in model.parameters() if p.requires_grad]
    assert [n for n,p in model.named_parameters() if p.requires_grad]==list(start['targets'])
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.load_state_dict(start['optimizer']);optimizer.zero_grad(set_to_none=True)
    assert tree_digest(optimizer.state_dict())==tree_digest(start['optimizer'])
    state={k:v.cuda() if torch.is_tensor(v) else v for k,v in start['own_state'].items()};assert tree_digest(state)==tree_digest(start['own_state'])
    readout=model.core.readout;readout.event_previous=[x.cuda() for x in start['event_previous']]
    assert tree_digest(readout.event_previous)==tree_digest(start['event_previous']) and readout.reference_bytes()==90316800
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();torch.set_rng_state(start['rng_cpu']);torch.cuda.set_rng_state(start['rng_cuda'])
    lo,hi=start['next_scan'],target['next_scan'];assert (lo,hi)==(375,788)
    goldpath=source/(mode+'_train_frames.jsonl');gold=[json.loads(x) for x in goldpath.read_text().splitlines()]
    updates=list(start['updates']);losses=[];units=[];ordinary=False;ends=set(contract['window_end_indices']);eligible=set(contract['expected_update_indices']);_,stream=loader(training[lo:hi],cfg,2,303)
    with (out/(mode+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream,start=lo):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));assert meta['frame']==training[i]['frame']==gold[i]['training']['frame']
            readout.begin_frame(bool(contract['expected_refresh_reasons'][i]))
            pred,state,info=model.step(ego,state,meta,detach_state=False);assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            assert info['full_refresh']==contract['expected_refresh_reasons'][i];ordinary|=not bool(info['full_refresh'])
            row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],refresh=info['full_refresh'],optimizer_step=False)
            extra=dict(index=i,weighted_event_mac=int(readout.event_weighted.detach()),unit_spike_ac=int(readout.event_units.detach()),objective_reference_bytes=readout.reference_bytes(),task_state_bytes=info['cache_bytes']);units.append(readout.event_units)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));losses.append(loss);row['detection_loss']=float(loss.detach())
            if i in ends:
                if i in eligible:
                    assert losses and ordinary;objective,det,event=window_objective(losses,units,True);objective.backward()
                    for n,p in model.named_parameters():
                        if p.requires_grad:assert p.grad is not None and bool(torch.isfinite(p.grad).all()),n
                        else:assert p.grad is None,n
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()),optimizer_step=True)
                    extra.update(window_detection_loss=float(det.detach()),window_event_energy_ratio=float(event.detach()),window_scans=len(units),window_labels=len(losses))
                    optimizer.step();updates.append(meta['frame']);assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];units=[];ordinary=False;state=detach_state(state)
            actual=dict(training=row,event=extra)
            log.write(json.dumps(dict(index=i,actual=actual,expected=gold[i],exact=actual==gold[i]))+'\n')
            assert actual==gold[i],('Original event-training trajectory mismatch',mode,i)
            if (i-lo+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',mode=mode,scans=i-lo+1,total=hi-lo,updates=len(updates)))
    assert updates==target['updates'] and len(updates)==128 and not losses and not units
    actual_targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    checks=dict(event_references=tree_digest(readout.event_previous)==tree_digest(target['event_previous']),targets=tree_digest(actual_targets)==tree_digest(target['targets']),optimizer=tree_digest(optimizer.state_dict())==tree_digest(target['optimizer']),task_state=tree_digest(state)==tree_digest(target['own_state']),rng_cpu=torch.equal(torch.get_rng_state(),target['rng_cpu']),rng_cuda=torch.equal(torch.cuda.get_rng_state(),target['rng_cuda']),frozen_weights=frozen_digest(model)==frozen,bn_buffers=buffer_digest(model)==buffers)
    result=dict(mode=mode,scans=hi-lo,updates_replayed=64,start_scan=lo,end_scan=hi,start_checkpoint_sha256=digest(paths[64]),target_checkpoint_sha256=digest(paths[128]),checks=checks,target_hash=tree_digest(actual_targets),optimizer_hash=tree_digest(optimizer.state_dict()),task_state_hash=tree_digest(state),event_reference_hash=tree_digest(readout.event_previous),parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=2449664,flops=None,model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/(mode+'_report.json'),result);assert all(checks.values()),checks
    return result


def worker(a,out):
    base,cfg,rows,source,exports,cal,rc,identity=prepare_event(a,out);lp=Path(a.event_learning_source);lv=json.loads(Path(a.event_learning_verification).read_text());c=json.loads((lp/'contract.json').read_text())
    training=rows[768:2816];assert [x['frame'] for x in training]==c['train_frames']
    for filename in ('checkpoint_manifest.json','event_train_frames.jsonl'):assert digest(lp/filename)==lv['source_sha256'][filename]
    manifest={x['name']:x for x in json.loads((lp/'checkpoint_manifest.json').read_text())['files']}
    write_json(out/'contract.json',dict(**identity,source_manifest_sha256=digest(lp/'checkpoint_manifest.json'),start_update=64,end_update=128,start_scan=375,end_scan=788,frames=[x['frame'] for x in training[375:788]],boundary='Sole quality/training GPU, no AP/new candidate. Fixed original64->128 replay for the event-trained terminal, own AdamW/task/RNG restored, original global TBPTT schedule retained. Exact per-frame detection/total loss, gradient and event-unit ledger plus final targets/optimizer/task/sending references/RNG required; any mismatch is failure, no tolerance tuning. No model FLOPs/timing measured; training replay not deployment.'))
    results={}
    for mode in ('event',):
        model=make_model(source,cal,rc)
        results[mode]=replay(mode,model,cfg,training,lp,out,c,manifest);del model;torch.cuda.empty_cache()
    write_json(out/'report.json',dict(status='COMPLETED',results=results,original_checkpoint_replay_exact=True,ap_evaluated=False,timing_valid=False,new_candidate_created=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in EVENT_ARGUMENTS:p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min fixed event optimizer replay')));signal.alarm(5400)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
