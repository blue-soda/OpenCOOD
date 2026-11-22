"""Score static and temporally reused observation convolutions in the same frozen detector."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_nonzero_observation import NonzeroObservationConv
from opencood.models.sub_modules.int_shallow_pulse import ShallowPulseConv
from opencood.models.sub_modules.int_shallow_anchor import AnchoredObservationConv
from opencood.models.sub_modules.int_bounded_sender import BoundedObservationConv
from opencood.tools.eval_int_address_cross import (
    AddressWriter, BranchFusion, CachedResidualPolicy, EdgeView, GroupedEdge, InspectMiddle,
    ObservedPolicy, SignedProjectionAccumulator, SilencePolicy, TaskDrive, build, compact_group_bucket,
    conv_costs, digest, loader, make_trace, prepare_stream, read, run_step, score, spatial_input,
    to_device, write_json)


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    parent = read(a.cross/'contract.json')
    channel = a.channel_calibration is not None
    assert (parent['group']=='original' or (channel and parent['group']=='additional')) and read(a.cross/'report.json')['status']=='COMPLETED'
    assert parent['weights']==digest(a.fit/'adapters.pt') and parent['plan']==digest(a.plan)
    assert parent['quota']==digest(a.quota) and parent['drive']==digest(a.drive/'report.json')
    assert parent['frozen_source']==read(a.long_base/'inputs/r297.json')['source_frozen_digest']
    baseline = [json.loads(l) for l in (a.cross/'frames.jsonl').read_bytes().splitlines()]
    measured = baseline if channel else [json.loads(l) for l in (a.live/'frames.jsonl').read_bytes().splitlines()]
    exact = None
    kernel_records=None
    scales=None
    if channel:
        table=read(a.channel_calibration)
        assert table['status']=='COMPLETED' and table['source']==parent['frozen_source'] and table['plan']==parent['plan']
        assert table['labels']==table['optimizer_steps']==0
        if a.shallow=='full':
            assert parent['group']=='additional' and not (set(table['scenes']) & set(parent['scenes']))
        assert a.arm in ('anchor_float','anchor_absolute')
        if a.scale_rule in ('assigned','shuffled'):
            scales=table[a.scale_rule]
        elif a.scale_rule=='fine':
            scales=[.125]*128
        elif a.scale_rule in ('generic','event'):
            allocation=read(a.cost_calibration)
            assert allocation['status']=='COMPLETED' and allocation['calibration']==digest(a.channel_calibration)
            scales=allocation['assignments'][a.scale_rule]['scales']
        assert a.arm=='anchor_absolute' or a.scale_rule=='uniform'
        if a.channel_receiver!='legacy':
            kernel=read(a.kernel_trace/'contract.json')
            assert kernel['parent']==parent and kernel['calibration']==digest(a.channel_calibration)
            assert kernel['rule']==(a.bounded_sender or ('float' if a.arm=='anchor_float' else a.scale_rule))
            if a.cost_calibration:assert kernel['cost_calibration']==digest(a.cost_calibration)
            assert read(a.kernel_trace/'report.json')['status']=='COMPLETED'
            kernel_records=[json.loads(l) for l in (a.kernel_trace/'frames.jsonl').read_bytes().splitlines()]
    elif a.shallow:
        assert read(a.exact_pair/'report.json')['status']=='COMPLETED'
        assert read(a.exact_pair/'contract.json')['parent']==parent
        exact = [json.loads(l) for l in (a.exact_pair/'frames.jsonl').read_bytes().splitlines()]
    plan = read(a.plan)
    rows = [r for r in plan['rows']['fit'] if r['scene'] in parent['scenes']]
    if a.shallow=='engineering':
        rows = [r for r in rows if r['scene']==parent['scenes'][0]]
        measured,baseline = [records[:len(rows)] for records in (measured,baseline)]
        if exact is not None:exact=exact[:len(rows)]
    for r in rows:
        assert r['split']=='train'
        for key in ('pcd','label'):
            if r[key]:assert digest(Path(r[key]))==plan['members'][r[key]]
    source,cfg,_,_,_,_,q = prepare_stream(a.long_base/'inputs',a.long_base/'clips')
    source.eval().requires_grad_(False)
    core = source.core
    fusion = BranchFusion(core.readout.layers[0].double_conv[0][0]).eval()
    core.readout.layers[0].double_conv[0][0] = fusion
    original = core.observation[0].conv2
    conv = NonzeroObservationConv(original).eval()
    pulse = None
    if a.bounded_sender:
        pulse=BoundedObservationConv(original,history=a.bounded_sender=='history_escape',signed=a.channel_receiver=='signed').eval()
    elif a.shallow:
        pulse = (AnchoredObservationConv(original,scales=scales,signed=a.channel_receiver!='mac',
                    signed_rebuild=a.channel_receiver=='signed') if a.arm.startswith('anchor_') else
                 ShallowPulseConv(original,floating=a.arm=='float_current')).eval()
    core.observation[0].conv2 = conv
    visits = []
    hook = conv.register_forward_hook(lambda m,inputs,out:visits.append(m.visits))
    edge = GroupedEdge(source.bounds,16384,read(a.long_base/'book_train/report.json')['levels'])
    rates = read(a.quota)['models']['age']['rates']
    coeff = read(a.drive/'report.json')['fit']['coefficients']
    weights = torch.load(a.fit/'adapters.pt',map_location='cpu')['amplitude']
    names = (a.arm,) if a.shallow else ('static','temporal')
    models,writers = {},{}
    for n in names:
        model = build(source,q,EdgeView(edge,'tile'))
        gate = SilencePolicy(edge.arms['tile'],'silent')
        exe = InspectMiddle(False,model.coarse_executor.receive_fine,gate)
        projector = model.coarse_projector
        def grant(count,p=projector):
            return int(count*rates[str(p.age)]+.5)
        writer = AddressWriter(TaskDrive(coeff,'continuous'),exe,grant,'amplitude',False)
        writer.adapter.copy_(weights)
        exe.drive = writer
        projector.event_linear = SignedProjectionAccumulator(projector,False)
        projector.event_observer = writer.observe_events
        model.coarse_executor = exe
        models[n],writers[n] = model,writer
    costs = conv_costs(core)
    reasons = make_trace(models[names[0]],rows)
    ds,stream = loader(rows,cfg,0,303)
    assert len(rows)==len(measured)==len(baseline)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        parent=parent,live_trace=digest(a.live/'frames.jsonl') if a.live else None,dense_trace=digest(a.cross/'frames.jsonl'),
        adapter='amplitude',training=False,scenes=list(dict.fromkeys(r['scene'] for r in rows)),
        shallow=a.shallow,arm=a.arm, exact_pair=digest(a.exact_pair/'frames.jsonl') if a.exact_pair else None,
        quantum=scales if scales is not None else .25 if a.shallow and a.arm not in ('float_current','temporal','anchor_float') else None,
        accumulation_scale=scales if scales is not None else pulse.quantum if pulse is not None else None,
        scale_rule=a.scale_rule if channel else None, calibration=digest(a.channel_calibration) if channel else None,
        cost_calibration=digest(a.cost_calibration) if a.cost_calibration else None,
        channel_receiver=a.channel_receiver, kernel_trace=digest(a.kernel_trace/'frames.jsonl') if kernel_records else None,
        **(dict(bounded_sender=a.bounded_sender, sender_quantum=.25, sender_tolerance=1/16) if a.bounded_sender else {}),
        tile=4 if a.shallow and a.arm.startswith(('motion_','fixed_')) else None,
        fraction=.5 if a.shallow and a.arm.startswith(('motion_','fixed_')) else None))
    factory = partial(CachedResidualPolicy,compiler=compact_group_bucket)
    labels = 0
    with (a.output/'frames.jsonl').open('x',buffering=1) as file:
        for i,(sample,reason,ref,base) in enumerate(zip(stream,reasons,measured,baseline)):
            meta = sample['meta']
            assert ref['frame']==base['frame']==meta['frame']==rows[i]['frame']
            assert ref['refresh']==base['refresh']==reason
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                states = dict.fromkeys(names)
                selectors = {n:ObservedPolicy(models[n],factory) for n in names}
                check_next = True
            ego = to_device(sample['ego'],torch.device('cuda'))
            spatial = spatial_input(source,ego)
            edge.start(meta,spatial.shape[-2:],reason)
            gt = ds.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] and a.shallow!='engineering' else None
            labels += int(gt is not None)
            item = dict(frame=meta['frame'],scene=meta['scene'],refresh=reason,arms={})
            if gt is not None:item['points'] = base['points']
            first_pred = None
            for n in names:
                adapted = a.shallow and n!='temporal'
                if adapted:
                    core.observation[0].conv2 = pulse
                    pulse.begin(n,reason,meta,spatial.shape[-2:],source.bounds,a.shallow=='engineering' or check_next)
                else:
                    core.observation[0].conv2 = conv
                    conv.begin(n=='temporal',n=='temporal' and reason is not None)
                pred,states[n],rec = run_step(models[n],writers[n],selectors[n],fusion,spatial,meta,
                    states[n],writers[n].charge,reason,'A',None,costs)
                if adapted:
                    rec['shallow'] = pulse.record
                    rec['observation_mac'] = pulse.record['rebuild_mac']+pulse.record['delta_mac']+pulse.record.get('reference_mac',0)
                    if kernel_records is not None:
                        assert kernel_records[i]['frame']==meta['frame']
                        cached=kernel_records[i]['arms']['temporal_'+a.channel_receiver]
                        assert all(cached[k]==v for k,v in pulse.record.items())
                else:
                    assert len(visits)==1 and visits[0]==ref[n]
                    rec['observation_mac'] = visits.pop()['mac']
                if gt is not None:rec.update(score(ds.post,ego,pred,gt,record_matches=True))
                if not a.shallow:
                    if first_pred is None:first_pred = pred
                    else:assert all(torch.equal(v,first_pred[k]) for k,v in pred.items())
                item['arms'][n] = rec
            if a.shallow:
                if exact is not None:
                    expected = exact[i]['arms']['temporal']
                    assert exact[i]['frame']==meta['frame']
                    if a.arm=='temporal':
                        actual = json.loads(json.dumps(item['arms']['temporal'],allow_nan=False))
                        assert all(v==expected[k] for k,v in actual.items()),(meta['frame'],[k for k,v in actual.items() if v!=expected[k]])
                if channel and gt is not None:assert len(item['points'])==rec['ap'][.7]['gt']
                if reason is None:check_next=False
            else:
                x,y = item['arms'].values()
                assert {k:v for k,v in x.items() if k!='observation_mac'}=={k:v for k,v in y.items() if k!='observation_mac'}
            file.write(json.dumps(item,allow_nan=False)+'\n')
    hook.remove()
    assert i+1==len(rows)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,labels=labels,optimizer_steps=0))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    for key in ('long-base','plan','quota','drive','fit','cross','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--live',type=Path)
    p.add_argument('--channel-calibration',type=Path)
    p.add_argument('--scale-rule',choices=('uniform','assigned','shuffled','fine','generic','event'),default='uniform')
    p.add_argument('--cost-calibration',type=Path)
    p.add_argument('--channel-receiver',choices=('legacy','signed','mac'),default='legacy')
    p.add_argument('--kernel-trace',type=Path)
    p.add_argument('--bounded-sender',choices=('history_escape','current_escape'))
    p.add_argument('--shallow',choices=('engineering','full'))
    p.add_argument('--arm',choices=('temporal','float_current','instant','fixed_amplitude','fixed_unit','motion_amplitude','motion_unit',
                                    'anchor_float','anchor_absolute','anchor_relative'))
    p.add_argument('--exact-pair',type=Path)
    args=p.parse_args()
    if args.bounded_sender and (args.arm!='anchor_float' or args.channel_receiver=='legacy' or not args.kernel_trace):
        p.error('bounded sender requires the anchor_float receiver and matching kernel trace')
    if args.channel_calibration and not args.shallow:p.error('channel evaluation requires an engineering or scoring phase')
    if args.scale_rule in ('generic','event') and not args.cost_calibration:p.error('cost-aware rules require a locked allocation')
    if args.channel_receiver!='legacy' and not (args.channel_calibration and args.kernel_trace):
        p.error('matched receiver evaluation requires calibration and a receiver trace')
    if not args.channel_calibration and not args.live:p.error('original evaluation requires the live convolution receipt')
    if args.shallow and not (args.arm and (args.exact_pair or args.channel_calibration)):p.error('missing shallow evaluation provenance')
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)
