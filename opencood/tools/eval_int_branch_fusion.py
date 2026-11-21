"""Compare shared three-branch fusion capability on frozen online detectors."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_branch_fusion import BranchFusion
from opencood.models.sub_modules.int_direct_fusion import DirectFusion
from opencood.models.sub_modules.int_compact_middle_update import conv_costs
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fusion_lattice import FusionInputLattice
from opencood.models.sub_modules.int_silent_receiver import PairedEdge, EdgeView
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_online import CachedResidualPolicy, ObservedPolicy, local_model
from opencood.tools.eval_int_fusion_cache import complete_scenes
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.eval_int_online_half import build
from opencood.tools.eval_int_codebook_fraction import FullAmplitude
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import compact_group_bucket, spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.run_int_silent_receiver import training_rows


@torch.no_grad()
def run(args):
    base = args.long_base
    args.inputs = base/'inputs'
    source,cfg,short,_,_,_,q = prepare_stream(args.inputs,base/'clips')
    book_path = base/'book_train/report.json'
    book = read(book_path)
    assert book['cap']==428149 and book['levels']==[3,6,13,28]
    audit = args.partition=='train'
    rows,traces,hashes = (training_rows(base,book_path,True) if audit else
        complete_scenes(args,short,remainder=args.partition=='transfer'))
    fusion = source.core.readout.layers[0].double_conv[0]
    grid = read(args.quanta) if args.quanta else None
    if grid:
        assert grid['status']=='COMPLETED' and grid['scans']==211 and grid['labels']==0
    layer = (DirectFusion(fusion[0],grid['quanta'],audit,args.weighted_check) if args.direct else
             BranchFusion(fusion[0],verify=audit)).eval()
    fusion[0] = layer
    probe = None
    if args.collect_rms or (args.quanta and not args.direct):
        probe=FusionInputLattice(grid['quanta'] if grid else None)
        layer.register_forward_pre_hook(probe.observe)
    edge = paired = None
    if args.route=='ann':
        model = local_model(source,indexed=True)
    elif args.route in ('half','amplitude'):
        edge = (FineCodebook(source.bounds,book['cap']//2,book['levels']) if args.route=='half' else
                FullAmplitude(source.bounds,'amplitude',book['cap']//2))
        model = build(source,q,edge)
    else:
        assert read(args.calibration/'report.json')['threshold']==12
        paired = PairedEdge(source.bounds,book['cap']//2,book['levels'],[12],calibration=True)
        model = build(source,q,EdgeView(paired,'12'))
    costs = conv_costs(source.core)
    refresh_audit=None
    if args.refresh_audit:
        from opencood.tools.audit_int_refresh_projection import RefreshProjectionAudit
        refresh_audit=RefreshProjectionAudit()
    pillar_macs=[0]
    def count_pillar(module,inputs,output):
        pillar_macs[0]+=output.numel()*module.in_features
    pillar_hooks=([module.register_forward_hook(count_pillar) for module in source.pillar_vfe.modules()
                   if isinstance(module,torch.nn.Linear)] if args.arithmetic_only else [])
    data, stream = loader(rows,cfg,0,303)
    historic = None
    arm_name = 'amplitude' if args.route=='amplitude' else 'carry'
    if args.route in ('half','amplitude') and not audit:
        part='remainder' if args.partition=='transfer' else 'dev'
        historic=[json.loads(s) for s in (base/('half_online_'+part)/'frames.jsonl').read_bytes().splitlines()]
        assert len(historic)==len(rows)
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        route=args.route,partition=args.partition,input_hashes=hashes,book=digest(book_path),seed=303,
        calibration=digest(args.calibration/'report.json') if paired else None,
        scenes=sorted({r['scene'] for r in rows}),params=sum(p.numel() for p in source.parameters()),
        collect_rms=args.collect_rms,quanta=digest(args.quanta) if args.quanta else None,
        direct=args.direct,weighted_check=args.weighted_check,arithmetic_only=args.arithmetic_only))
    ordinary=labels=0
    with (args.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace) in enumerate(zip(stream,traces)):
            meta=sample['meta']
            assert meta['frame']==trace['frame']
            reason=trace['refresh']
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                prev=None if reason=='initial' else dict(scene='')
                selector=ObservedPolicy(model,partial(CachedResidualPolicy,compiler=compact_group_bucket))
            ego=to_device(sample['ego'],torch.device('cuda'))
            pillar_macs[0]=0
            spatial=spatial_input(source,ego)
            assert model.refresh_reason(prev,meta)==reason
            if paired: paired.start(meta,spatial.shape[-2:],reason)
            if edge: edge.begin(prev,meta,spatial.shape[-2:],reason)
            if args.route!='ann': model.coarse_projector.begin(prev,reason)
            selector.observation=None
            layer.begin(prev,reason)
            if probe:probe.begin(reason)
            (pred,state,info),other=measured_step(model,spatial,prev,meta,selector)
            assert info['full_refresh']==reason
            record=dict(frame=meta['frame'],scene=meta['scene'],segment=meta['segment'],refresh=reason,fusion=layer.ledger)
            if refresh_audit is not None and reason is not None:
                record['refresh_projection']=refresh_audit.measure(model,state,prev,reason,meta,spatial.shape[-2:])
            if args.arithmetic_only:
                components=dict(pillar_mac=pillar_macs[0],merge_add=layer.ledger['merge_add'])
                for name,projector in [('middle',model.coarse_projector),('deep',model.deep_projector)]:
                    v=projector.ledger
                    assert bool(v['rebuilt'])==(reason is not None) and projector.mode=='amplitude'
                    components[name]=dict(fp64_mac=v['dense_mac'] if v['rebuilt'] else 0,
                                          integer_mac=0 if v['rebuilt'] else v['amplitude_mac'],
                                          unit_ac_if_expanded=0 if v['rebuilt'] else v['unit_ac'],
                                          decode_mul=v['dense_mac']//projector.weight.shape[0],
                                          decode_bias=projector.block[0].bias is not None)
                taps=info['edge'] if reason is None else {}
                components['depthwise_integer_mac']=taps.get('cold_nonzero_taps',0)+taps.get('delta_nonzero_taps',0)
                components['depthwise_decode_ops']=2*taps.get('sites',0)*model.stage.edge.weight.shape[0]
                record['arithmetic']=components
            if args.quanta and probe:record['lattice']=probe.ledger
            if reason is None:
                ordinary+=1
                if args.route!='ann':
                    exe=model.coarse_executor
                    other+=sum(costs[k]*v for k,v in exe.counts.items())
                    exe.inputs=None
                written=(state['deep_age']==0).flatten().nonzero().flatten().tolist()
                if args.route=='ann' and not audit: assert written==trace['written_indices']
                if historic: assert written==historic[i]['arms'][arm_name]['written']
                if args.route=='amplitude':
                    record['addresses']=edge.ledger['addresses']
                    record['payload_bytes']=edge.ledger['bytes']
                    assert record['addresses']==historic[i]['arms'][arm_name]['addresses']
            if edge: edge.commit(state,False)
            state['branch_fusion']=layer.new
            prev=state
            if args.direct:
                counts=layer.ledger['branches']
                ac=sum(b['one_static']+b['one_delta'] for b in counts)*128
                mac=sum(b['static_chosen']+b['delta_chosen'] for b in counts)*128-ac
                record.update(other=other,fusion_ac=ac,fusion_mac=mac,core=other+2*mac+ac)
            else:
                site=layer.ledger['sites']
                deep_taps=layer.ledger['branches'][2]['executed_taps']
                original_fusion=(site*384*9*256 if args.route=='ann' else site*256*9*256+deep_taps*256)
                record.update(other=other,legacy_core=other+original_fusion,core=other+layer.ledger['float_conv'])
            if not audit:
                old=(trace['arms']['indexed_crop_online']['core_conv_flops'] if args.route=='ann'
                     else historic[i]['arms'][arm_name]['core'])
                if not args.quanta:
                    assert old==record['legacy_core'],(meta['frame'],old,record['legacy_core'])
            if meta['supervised'] and not args.arithmetic_only:
                labels+=1
                gt=data.post.generate_gt_bbx(dict(ego=ego))
                record.update(score(data.post,ego,pred,gt,record_matches=True))
            log.write(json.dumps(record)+'\n')
    expected={'train':(211,0),'dev':(1014,458),'transfer':(4780,1280)}
    size,expected_labels=expected[args.partition]
    assert (i+1,labels)==(size,0 if args.arithmetic_only else expected_labels)
    for hook in pillar_hooks:hook.remove()
    report=dict(status='COMPLETED',scans=len(rows),ordinary=ordinary,labels=labels,optimizer_steps=0)
    if args.collect_rms:
        report.update(quanta=probe.calibrated(),samples_per_channel=probe.samples)
    write_json(args.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--route',choices=['ann','half','amplitude','threshold'],required=True)
    p.add_argument('--partition',choices=['train','dev','transfer'],required=True)
    p.add_argument('--collect-rms',action='store_true')
    p.add_argument('--direct',action='store_true')
    p.add_argument('--weighted-check',action='store_true')
    p.add_argument('--arithmetic-only',action='store_true')
    p.add_argument('--refresh-audit',action='store_true')
    p.add_argument('--quanta',type=Path)
    for k in ('long-base','vehicle','calibration','output'):
        p.add_argument('--'+k,type=Path,required=k in ('long-base','output'))
    a=p.parse_args()
    if a.refresh_audit and (not a.arithmetic_only or a.partition!='train'):
        p.error('refresh projection checks use the frozen training-side arithmetic replay')
    if a.arithmetic_only and (a.route!='half' or a.quanta or a.collect_rms):
        p.error('arithmetic replay requires the original half-budget detector')
    if a.route=='amplitude' and a.partition=='train':p.error('use the historical paired stream for amplitude')
    if a.direct and not a.quanta:p.error('direct signed fusion needs a frozen lattice')
    if a.weighted_check and (not a.direct or a.partition!='train'):
        p.error('the weighted control is a training-side direct fusion audit')
    if (a.collect_rms or a.quanta) and a.route!='half':p.error('fusion lattice requires the frozen half route')
    if a.collect_rms and (a.partition!='train' or a.quanta):p.error('RMS calibration requires raw training projections')
    if a.route=='threshold' and (a.partition!='train' or a.calibration is None):
        p.error('threshold route is the training-side engineering audit')
    if a.partition!='train' and a.vehicle is None:p.error('historical evaluation requires vehicle data')
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
