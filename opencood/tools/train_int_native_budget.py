"""Fit task membranes inside a deployable native event budget, on disjoint adapter splits."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_native_silent_events import NativeEventWriter, SignedProjectionAccumulator
from opencood.tools.train_int_silent_task import (
    BranchFusion, CachedResidualPolicy, EdgeView, GroupedEdge, InspectMiddle, ObservedPolicy,
    SilencePolicy, TaskDrive, block_lidar_range, build, compact_group_bucket, conv_costs, digest,
    loader, measured_step, prepare_stream, read, readout_loss, score, spatial_input, to_device, write_json)
from opencood.utils import box_utils, pcd_utils


def gradients(writer, exe, previous, area, past_pose, pose):
    """Fixed delivered support STE, with the same forward-zero tangent in both modes."""
    if previous is None:
        moved = torch.zeros_like(exe.result)
        charge = torch.zeros_like(moved)
    else:
        moved, charge = [warp_bev(v,past_pose,pose,area)[0] for v in previous]
    drive = writer.drive
    ids, old, shallow, base = drive.saved
    old_grad = moved[0].flatten(1)[:,ids].T
    base_grad = old_grad*(1+drive.weight[:,0])*(base>0)
    proposal = (base + base_grad + drive.correction(old+old_grad,shallow)).clamp_min(0) - (old+old_grad)
    total = charge[0].flatten(1)[:,ids] + (proposal-proposal.detach()).T
    sent = total * writer.selected
    out = moved*exe.skip
    out[0].flatten(1)[:,ids] = old_grad.T + sent
    remainder = charge*exe.skip
    remainder[0].flatten(1)[:,ids] = total-sent
    return out,remainder


def make_trace(model,rows):
    trace=[]
    state=None
    for i,r in enumerate(rows):
        if i==0 or r['segment']!=rows[i-1]['segment']:
            state=None
        reason=model.refresh_reason(state,r)
        stamp=int(r['timestamp_us'])
        state=dict(scene=r['segment'], timestamp_us=stamp, since_refresh=0 if reason else state['since_refresh']+1,
                   refresh_timestamp_us=stamp if reason else state['refresh_timestamp_us'])
        trace.append(reason)
    return trace


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    train=a.mode=='fit'
    source,cfg,short,_,_,_,q=prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    core=source.core
    plan=read(a.plan)
    identity=read(a.plan.parent/'report.json')
    assert digest(a.plan)==identity['plan'] and identity['status']=='COMPLETED'
    split='fit' if train else 'internal'
    rows=plan['rows'][split]
    for r in rows:
        for key in ('pcd','label'):
            if r[key]:assert digest(Path(r[key]))==plan['members'][r[key]]
    assert all(r['split']=='train' for r in rows)
    totals=identity['splits'][split]
    historical_traces=None
    if a.mode=='dev':
        from opencood.tools.eval_int_fusion_cache import complete_scenes
        a.inputs=a.long_base/'inputs'
        rows,historical_traces,_=complete_scenes(a,short,remainder=False)
        split='dev'
        totals=dict(scans=1014,labels=458,scenes=list(dict.fromkeys(r['scene'] for r in rows)))
    calibration=read(a.quota)
    assert calibration['selected']=='age'
    rates=calibration['models']['age']['rates']
    frozen_fit=read(a.drive/'report.json')['fit']['coefficients']
    book=read(a.long_base/'book_train/report.json')
    edge=GroupedEdge(source.bounds,16384,book['levels'])
    dense=copy.deepcopy(core.readout) if train else None
    block=core.readout.layers[0].double_conv[0]
    fusion=BranchFusion(block[0]).eval()
    block[0]=fusion
    captured=[]
    if train:
        hook=core.readout.register_forward_pre_hook(lambda layer,inputs:captured.append(inputs[0].detach()))
    names=('unit','amplitude') if train else ('unit_zero','amplitude_zero','unit','amplitude')
    weights=None
    if not train:
        receipt=read(a.fit/'report.json')
        assert receipt['status']=='COMPLETED' and receipt['weights']==digest(a.fit/'adapters.pt')
        assert read(a.fit/'contract.json')['plan']==identity['plan']
        assert read(a.fit/'contract.json')['quota']==digest(a.quota)
        weights=torch.load(a.fit/'adapters.pt',map_location='cpu')
    models, writers, opts={},{},{}
    for n in names:
        mode='amplitude' if n.startswith('amplitude') else 'integrate'
        model=build(source,q,EdgeView(edge,'tile'))
        policy=SilencePolicy(edge.arms['tile'],'silent')
        exe=InspectMiddle(False,model.coarse_executor.receive_fine,policy)
        projector=model.coarse_projector
        def grant(count, projection=projector):
            return int(count*rates[str(projection.age)]+.5)
        writer=NativeEventWriter(TaskDrive(frozen_fit,'continuous'),exe,grant,mode,False)
        writer.record_training_mask=train
        if train:
            opts[n]=torch.optim.Adam([writer.adapter],lr=.005)
        elif not n.endswith('_zero'):
            writer.adapter.copy_(weights[n])
        exe.drive=writer
        model.coarse_executor=exe
        projector.event_observer=writer.observe_events
        projector.event_linear=SignedProjectionAccumulator(projector,False)
        models[n],writers[n]=model,writer
    contract=dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                  split=split,plan=identity['plan'],quota=digest(a.quota),rates=rates,seed=303,
                  params=sum(p.numel() for p in source.parameters()),adapter=2560,frozen_drive=2560,
                  drive=digest(a.drive/'report.json'),frozen_source=read(a.long_base/'inputs/r297.json')['source_frozen_digest'],
                  book=digest(a.long_base/'book_train/report.json'),scenes=totals['scenes'],
                  optimizer='Adam lr=.005, clip=5, one ordered pass, identical 2-frame ordinary windows',
                  backward='fixed selected units, direct middle projection STE, pose-aligned state and residual',
                  fits_parent=digest(a.fit/'adapters.pt') if not train else None)
    write_json(a.output/'contract.json',contract)
    reasons=(make_trace(models[names[0]],rows) if historical_traces is None else [r['refresh'] for r in historical_traces])
    ds,stream=loader(rows,cfg,0,303)
    costs=conv_costs(core)
    factory=partial(CachedResidualPolicy,compiler=compact_group_bucket)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda().eval() if train else None
    steps=dict.fromkeys(names,0)
    cross={n:[0,0] for n in names}
    labels=ordinary_labels=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,reason) in enumerate(zip(stream,reasons)):
            meta=sample['meta']
            assert meta['frame']==rows[i]['frame']
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                states=dict.fromkeys(names,None if reason=='initial' else dict(scene=''))
                selectors={n:ObservedPolicy(models[n],factory) for n in names}
            if reason:
                depth=0
                tangents=dict.fromkeys(names)
                previous_sites=dict.fromkeys(names)
                losses={n:[] for n in names}
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            edge.start(meta,spatial.shape[-2:],reason)
            labelled=meta['supervised']
            labels+=int(labelled)
            ordinary_labels+=int(labelled and reason is None)
            end=reason is None and (depth==1 or i+1==len(reasons) or reasons[i+1] is not None)
            item=dict(frame=meta['frame'],scene=meta['scene'],refresh=reason,arms={})
            gt=ds.post.generate_gt_bbx(dict(ego=ego)) if labelled and not train else None
            if gt is not None:
                cloud,_=pcd_utils.read_pcd(rows[i]['pcd'])
                cloud=pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(cloud,cfg['preprocess']['cav_lidar_range']))[:,:3]
                item['points']=[len(box_utils.get_points_in_rotated_box_3d(cloud,b)) for b in gt.cpu().numpy()]
            masks=None
            for n in names:
                model,writer=models[n],writers[n]
                prev=states[n]
                assert model.refresh_reason(prev,meta)==reason
                model.coarse_projector.begin(prev,reason)
                fusion.begin(prev,reason)
                writer.begin_frame(prev,meta,spatial.shape[-2:],source.bounds,reason)
                selectors[n].observation=None
                (pred,state,info),core_ops=measured_step(model,spatial,prev,meta,selectors[n])
                assert info['full_refresh']==reason
                state['branch_fusion']=fusion.new
                states[n]=state
                rec=dict(core=core_ops+fusion.ledger['float_conv'])
                if train:
                    assert len(captured)==1
                    branches=captured.pop()
                if reason is None:
                    exe=model.coarse_executor
                    if masks is None:masks=(exe.skip,exe.refine_skip.extra)
                    else:assert torch.equal(masks[0],exe.skip) and torch.equal(masks[1],exe.refine_skip.extra)
                    rec.update(native=writer.native,projection=model.coarse_projector.event_linear.stats)
                    rec['core']+=exe.drive_ops+sum(costs[k]*v for k,v in exe.counts.items())
                    if train:
                        with torch.enable_grad():
                            past=tangents[n]
                            area=block_lidar_range(source.bounds,spatial.shape[-2:],exe.result.shape[-2:],4)
                            now=gradients(writer,exe,past,area,prev['pose'],meta['pose'])
                            if labelled:
                                loss=readout_loss(core,dense,state['coarse'],now[0],branches,pred,criterion,ego['label_dict'])
                                losses[n].append(loss)
                                rec['loss']=float(loss.detach())
                                if past is not None:
                                    g=torch.autograd.grad(loss,past,retain_graph=True)
                                    values=[float(v.flatten(2)[:,:,previous_sites[n]].abs().sum()) for v in g]
                                    rec['previous_middle_charge_grad_l1']=values
                                    for j,value in enumerate(values):cross[n][j]+=int(value>0)
                            tangents[n]=now
                            previous_sites[n]=writer.drive.saved[0]
                            if end and losses[n]:
                                opts[n].zero_grad(set_to_none=True)
                                torch.stack(losses[n]).mean().backward()
                                norm=torch.nn.utils.clip_grad_norm_([writer.adapter],5.)
                                assert torch.isfinite(norm)
                                opts[n].step()
                                steps[n]+=1
                                rec['grad_norm']=float(norm)
                        if end:
                            tangents[n]=None
                            losses[n].clear()
                    exe.inputs=exe.result=None
                if gt is not None:rec.update(score(ds.post,ego,pred,gt,record_matches=True))
                item['arms'][n]=rec
            if reason is None:depth=0 if end else depth+1
            item['wire_bytes']=edge.stats['tile']['bytes']
            log.write(json.dumps(item,allow_nan=False)+'\n')
            if (i+1)%50==0:
                write_json(a.output/'status.json',dict(scans=i+1,steps=steps,previous_gradients=cross))
    assert len(rows)==i+1==totals['scans'] and labels==totals['labels']
    report=dict(status='COMPLETED',scans=i+1,labels=labels,ordinary_labels=ordinary_labels,
                steps=steps,previous_gradients=cross)
    if train:
        hook.remove()
        assert steps['amplitude']==steps['unit']>0 and all(min(v)>0 for v in cross.values())
        torch.save({n:w.adapter.detach().cpu() for n,w in writers.items()},a.output/'adapters.pt')
        report['weights']=digest(a.output/'adapters.pt')
    write_json(a.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('fit','internal','dev'))
    for k in ('long-base','plan','drive','quota','fit','output'):
        p.add_argument('--'+k,type=Path,required=k!='fit')
    p.add_argument('--vehicle',type=Path)
    p.add_argument('--internal-verdict',type=Path)
    a=p.parse_args()
    if a.mode=='dev':
        if not a.vehicle or not a.internal_verdict or read(a.internal_verdict).get('promote') is not True:
            p.error('development requires an accepted frozen internal comparison')
    if a.mode!='fit' and not a.fit:p.error('internal evaluation requires the frozen fit directory')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
