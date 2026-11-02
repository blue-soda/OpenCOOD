"""Matched retained/dropped subthreshold task-state learning, one quality GPU."""
from opencood.tools.pilot_int_causal_readout_qat import *
from opencood.tools.pilot_int_gated_state import evaluate
from opencood.tools.pilot_int_protected_state import schedule
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector
from opencood.tools.int_artifact_paths import artifact_path
PREFIX='core.coarse_update.'


def frozen_digest(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if not n.startswith(PREFIX):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def detach_state(state,cpu=False):
    return {k:(v.detach().cpu() if cpu else v.detach()) if torch.is_tensor(v) else v for k,v in state.items()}


def expected_updates(rows):
    ends=window_ends(rows,4);has_label=False;result=[]
    for i,r in enumerate(rows):
        has_label|=bool(r['supervised'])
        if i in ends:
            if has_label:result.append(r['frame'])
            has_label=False
    return result


def train_one(model,cfg,rows,out,name,criterion,contract,preflight=False):
    original_frozen=frozen_digest(model);before=model_digest(model);before_buffers=buffer_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad==n.startswith(PREFIX) for n,p in model.named_parameters())
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=None;losses=[];updates=[];loss_values=[];labels=0
    _,stream=loader(rows,cfg,2,303);tag=name+('_preflight' if preflight else '_train')
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
            assert meta['frame']==rows[i]['frame']
            pred,state,info=model.step(ego,state,meta,detach_state=False)
            assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],refresh=info['full_refresh'],optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss))
                losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if losses and meta['frame'] in contract['expected_update_frames']:
                    objective=torch.stack(losses).mean();objective.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(PREFIX))
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in opened)
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()))
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==before
                        model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],scans=i+1,gradient_norm=float(norm),weights_unchanged=True)
                    optimizer.step();updates.append(meta['frame']);loss_values.append(float(objective.detach()));row['optimizer_step']=True
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%16==0:
                    path=out/('%s_update_%03d.pth'%(name,len(updates)))
                    torch.save(dict(coarse=model.core.coarse_update.state_dict(),optimizer=optimizer.state_dict(),
                        own_state=detach_state(state,cpu=True),next_scan=i+1,updates=updates,contract=contract,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),mode=name),path)
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows),updates=len(updates)))
    assert not preflight and updates==contract['expected_update_frames'] and labels==232
    assert frozen_digest(model)==original_frozen and model_digest(model)!=before and buffer_digest(model)==before_buffers
    path=out/(name+'_coarse_final.pth')
    torch.save(dict(coarse=model.core.coarse_update.state_dict(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
        resume='Fixed pass complete; any continuation separately registered, start from empty stream state'),path)
    return dict(scans=512,labels=labels,updates=len(updates),update_frames=updates,frozen_weights_bn_unchanged=True,
        checkpoint_sha256=digest(path),all_buffers_unchanged=True,trainable_parameters=sum(p.numel() for p in opened),mean_window_loss=sum(loss_values)/len(loss_values))


def buffer_digest(model):
    h=hashlib.sha256()
    for n,v in model.named_buffers():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.full else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:
        path=Path(artifact_path(r['path'])) if a.full else Path(r['relocated_path']);assert digest(path)==r['sha256']
    a.mode='gru';a.stage='full' if a.full else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    ev=json.loads(Path(a.engineering_verification).read_text());assert ev['status']=='VERIFIED' and ev['run']=='R-191';ep=Path(a.engineering_source)/'contract.json';assert digest(ep)==ev['source_sha256']['contract.json'];ec=json.loads(ep.read_text());assert ec['checkpoint_sha256']==gate['checkpoint_sha256'] and ec['initialization_sha256']==cv['rank128_checkpoint_sha256'] and ec['quantum']==.25
    rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED' and rv['run']=='R-182';rp=Path(a.reference_source)/'ap_records.json';assert digest(rp)==rv['source_sha256']['ap_records.json'];gold=json.loads(rp.read_text())['control']
    models=dict(frozen=source);models.update({n:SubthresholdMembraneDetector(source,'continuous' if n=='original' else n,.25).cuda().eval() for n in ('original','carry','drop','reset')})
    for n,m in models.items():
        for name,p in m.named_parameters():p.requires_grad_(n!='frozen' and name.startswith(PREFIX))
    assert len({model_digest(m) for m in models.values()})==1
    training,evaluation=rows[:512],rows[512:768];reasons,_=schedule(training);ends=window_ends(training,4);ordinary=False;eligible=False;updates=[]
    for i,row in enumerate(training):
        ordinary|=not bool(reasons[i]);eligible|=bool(row['supervised']) and ordinary
        if i in ends:
            if eligible:updates.append(row['frame'])
            ordinary=eligible=False
    assert len(updates)>0 and set(updates).issubset(expected_updates(training))
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),initialization_sha256=digest(Path(a.initialization)),engineering_verification_sha256=digest(Path(a.engineering_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],expected_refresh_reasons=reasons,expected_update_frames=updates,skipped_label_window_frames=[f for f in expected_updates(training) if f not in updates],seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,quantum=.25,trainable_prefix=PREFIX,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),loss='Same labelled detection-window mean, no teacher loss/history; only ordinary coarse update learns. Freeze refresh candidates, readout, fine/deep updates, observations, heads and all BN statistics.',boundary='Fixed512/232 labels, all four arms same metadata-derived windows: optimize only if an ordinary frame occurs at or before a labelled frame within the4-frame window. No loss-dependent window selection. Continuous, retain subthreshold v, discard residual Q(v), coarse reset only; all same parameters/state budget. Initial/final all retained, no best epoch or quantum search. Surrogate gradient for quantizer; not sparse SNN training or independent task memory benefit by construction.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();names=('original','carry','drop','reset')
    preflight={n:train_one(models[n],cfg,training,out,n,criterion,contract,True) for n in names};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial',True);saved0=json.loads((out/'initial_ap_records.json').read_text());assert saved0['frozen']==saved0['original']==gold
    learned={n:train_one(models[n],cfg,training,out,n,criterion,contract) for n in names};final,raw1=evaluate(models,base,cfg,evaluation,out,'final');assert raw0['frozen']==raw1['frozen']
    write_json(out/'checkpoint_manifest.json',dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('*.pth'))],resume_rehearsed=False))
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,trained=True,timing_valid=False,frozen_raw_ap_exact=True,initial_ann_raw_ap_exact=True));write_json(out/'status.json',dict(status='COMPLETED'))

def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','engineering-verification','engineering-source','reference-verification','reference-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min matched subthreshold state adaptation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
