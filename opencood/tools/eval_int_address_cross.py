"""Single-scan address and payload interventions on two frozen causal BEV trajectories."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.models.sub_modules.int_native_silent_events import NativeEventWriter, SignedProjectionAccumulator
from opencood.tools.train_int_native_budget import make_trace
from opencood.tools.train_int_silent_task import (
    BranchFusion, CachedResidualPolicy, EdgeView, GroupedEdge, InspectMiddle, ObservedPolicy,
    SilencePolicy, TaskDrive, build, compact_group_bucket, conv_costs, digest, loader,
    measured_step, prepare_stream, read, score, spatial_input, to_device, write_json)
from opencood.utils import box_utils, pcd_utils


MODES = ('U', 'A')


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class AddressWriter(NativeEventWriter):
    forced = None

    def observe_events(self, events):
        ids = self.executor.refine_skip.extra.flatten().nonzero().flatten()
        values = events[0].flatten(1)[:, ids].reshape(-1)
        support = values.nonzero().flatten()
        self.received = dict(support=tensor_hash(support),
                             signs=tensor_hash(values[support].sign().to(torch.int8)))
        super().observe_events(events)

    def select_addresses(self, valid, total, grant):
        if self.forced is None:
            selected = super().select_addresses(valid,total,grant)
            self.native_ids = selected
            self.proposal = total
            self.feasible = valid.flatten()
            return selected
        assert torch.equal(total,self.proposal) and bool(valid.flatten()[self.forced].all())
        return self.forced


def matched_lists(writers):
    first,second = (writers[k] for k in MODES)
    for w in (first,second):
        assert torch.equal(w.drive.saved[0],first.drive.saved[0])
    shared = first.feasible & second.feasible
    ordered = {}
    for k,w in writers.items():
        ids = w.native_ids
        # Reproduce the native priority even when all eligible addresses fitted the grant.
        common = ids[shared[ids]]
        flat_ids = common.cpu().numpy()
        strength = w.proposal.flatten()[common].abs().cpu().numpy()
        order = torch.as_tensor(np.lexsort((flat_ids,-strength)),device=ids.device)
        ordered[k] = common[order]
    count = min(map(len,ordered.values()))
    trimmed = {k:v[:count] for k,v in ordered.items()}
    atlas = torch.zeros_like(shared)
    atlas[first.native_ids] = True
    overlap = int(atlas[second.native_ids].sum())
    atlas.zero_()
    atlas[trimmed['U']] = True
    return trimmed, dict(native={k:len(w.native_ids) for k,w in writers.items()},
                         mutually_legal={k:len(v) for k,v in ordered.items()}, retained=count,
                         native_overlap=overlap,matched_overlap=int(atlas[trimmed['A']].sum()),
                         quiet_hash=tensor_hash(first.drive.saved[0]),
                         native_hash={k:tensor_hash(w.native_ids.sort().values) for k,w in writers.items()},
                         matched_hash={k:tensor_hash(v.sort().values) for k,v in trimmed.items()})


def run_step(model,writer,selector,fusion,spatial,meta,prev,old_charge,reason,mode,forced,costs,extra_cost=None):
    writer.mode = 'integrate' if mode=='U' else 'amplitude'
    writer.forced = forced
    writer.charge = old_charge
    assert model.refresh_reason(prev,meta)==reason
    model.coarse_projector.begin(prev,reason)
    fusion.begin(prev,reason)
    writer.begin_frame(prev,meta,spatial.shape[-2:],model.source.bounds,reason)
    selector.observation = None
    (pred,state,info),core = measured_step(model,spatial,prev,meta,selector)
    assert info['full_refresh']==reason
    state['branch_fusion'] = fusion.new
    rec = dict(core=core+fusion.ledger['float_conv'],ac=0,mac=0)
    if reason is None:
        exe = model.coarse_executor
        ops = model.coarse_projector.event_linear.stats
        rec.update(ac=ops['ac'],mac=ops['mac'],sent=writer.native['sent'],
                   units=writer.native['sent_units'])
        rec.update(writer.received)
        rec['core'] += exe.drive_ops + sum(costs[k]*v for k,v in exe.counts.items())
        exe.inputs=exe.result=None
    if extra_cost is not None:
        extra_cost['fusion_merge']=fusion.ledger['merge_add']
        for key,p in [('middle',model.coarse_projector),('deep',model.deep_projector)]:
            v=p.ledger
            assert bool(v['rebuilt'])==(reason is not None)
            assert not v.get('static_refresh',False)
            extra_cost[key+'_decode']=v['dense_mac']//p.weight.shape[0]*(1+int(p.block[0].bias is not None))
            if reason is not None:
                extra_cost[key+'_refresh']=2*v['dense_mac']
            elif key=='deep':
                extra_cost['deep_event']=2*v['amplitude_mac']
        if reason is None:
            edge=info['edge']
            extra_cost['depthwise']=2*(edge['cold_nonzero_taps']+edge['delta_nonzero_taps'])
            extra_cost['depthwise_decode']=2*edge['sites']*model.stage.edge.weight.shape[0]
    return pred,state,rec


@torch.no_grad()
def run(a):
    base=a.long_base
    torch.manual_seed(303)
    source,cfg,_,_,_,_,q=prepare_stream(base/'inputs',base/'clips')
    source.eval().requires_grad_(False)
    data=read(a.plan)
    identity=read(a.plan.parent/'report.json')
    receipt=read(a.fit/'report.json')
    trained=read(a.fit/'contract.json')
    assert digest(a.plan)==identity['plan']==trained['plan']
    assert receipt['weights']==digest(a.fit/'adapters.pt') and receipt['status']=='COMPLETED'
    assert digest(a.quota)==trained['quota'] and digest(a.drive/'report.json')==trained['drive']
    assert trained['frozen_source']==read(base/'inputs/r297.json')['source_frozen_digest']
    weights=torch.load(a.fit/'adapters.pt',map_location='cpu')
    old=set(identity['original_fit_scenes'])
    scenes=[s for s in identity['splits']['fit']['scenes'] if (s in old)==(a.group=='original')]
    assert len(scenes)==(8 if a.group=='original' else 16)
    rows=[r for r in data['rows']['fit'] if r['scene'] in scenes]
    for r in rows:
        for key in ('label','pcd'):
            if r[key]:assert digest(Path(r[key]))==data['members'][r[key]]
    coeff=read(a.drive/'report.json')['fit']['coefficients']
    rates=read(a.quota)['models']['age']['rates']
    core=source.core
    fusion=BranchFusion(core.readout.layers[0].double_conv[0][0]).eval()
    core.readout.layers[0].double_conv[0][0]=fusion
    edge=GroupedEdge(source.bounds,16384,read(base/'book_train/report.json')['levels'])
    models,writers={},{}
    for k in MODES:
        model=build(source,q,EdgeView(edge,'tile'))
        gate=SilencePolicy(edge.arms['tile'],'silent')
        exe=InspectMiddle(False,model.coarse_executor.receive_fine,gate)
        projector=model.coarse_projector
        def grant(n,p=projector):
            return int(n*rates[str(p.age)]+.5)
        writer=AddressWriter(TaskDrive(coeff,'continuous'),exe,grant,'integrate',False)
        writer.adapter.copy_(weights['unit' if k=='U' else 'amplitude'])
        exe.drive=writer
        projector.event_linear=SignedProjectionAccumulator(projector,False)
        projector.event_observer=writer.observe_events
        model.coarse_executor=exe
        models[k],writers[k]=model,writer
    costs=conv_costs(core)
    reasons=make_trace(models['U'],rows)
    ds,stream=loader(rows,cfg,0,303)
    factory=partial(CachedResidualPolicy,compiler=compact_group_bucket)
    write_json(a.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        group=a.group,scenes=scenes,plan=trained['plan'],weights=receipt['weights'],quota=trained['quota'],
        drive=trained['drive'],frozen_source=trained['frozen_source'],params=trained['params'],adapter=2560,
        training=False,seed=303,integer_fanout='actual middle projection, AC + 2 MAC'))
    labels=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,reason) in enumerate(zip(stream,reasons)):
            meta=sample['meta']
            if i==0 or meta['segment']!=rows[i-1]['segment']:
                states=dict.fromkeys(MODES)
                selectors={k:ObservedPolicy(models[k],factory) for k in MODES}
                check_replay=True
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            edge.start(meta,spatial.shape[-2:],reason)
            gt=ds.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            item=dict(scene=meta['scene'],frame=meta['frame'],refresh=reason,arms={})
            assert meta['frame']==rows[i]['frame']
            if gt is not None:
                cloud,_=pcd_utils.read_pcd(rows[i]['pcd'])
                cloud=pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(cloud,cfg['preprocess']['cav_lidar_range']))[:,:3]
                item['points']=[len(box_utils.get_points_in_rotated_box_3d(cloud,b)) for b in gt.cpu().numpy()]
                labels+=1
            prior_states=states.copy()
            prior_charge={k:w.charge for k,w in writers.items()}
            gold={}
            def execute(k,mode,ids):
                pred,state,rec=run_step(models[k],writers[k],selectors[k],fusion,spatial,meta,
                    prior_states[k],prior_charge[k],reason,mode,ids,costs)
                if gt is not None:rec.update(score(ds.post,ego,pred,gt,record_matches=True))
                return pred,state,rec
            for k in MODES:
                pred,states[k],rec=execute(k,k,None)
                item['arms'][k+k+'n']=rec
                if check_replay:gold[k]=pred
            charges={k:w.charge for k,w in writers.items()}
            if reason is None:
                trimmed,item['addresses']=matched_lists(writers)
                for k,w in writers.items():
                    own=w.native_ids
                    other='A' if k=='U' else 'U'
                    item['arms'][k+other+'n']=execute(k,other,own)[2]
                    for loc,ids in trimmed.items():
                        for mode in MODES:
                            key=k+loc+mode
                            item['arms'][key]=execute(k,mode,ids)[2]
                            assert item['arms'][key]['sent']==len(ids)
                    if check_replay:
                        pred,_,rec=execute(k,k,own)
                        assert all(torch.equal(v,gold[k][key]) for key,v in pred.items())
                        assert rec==item['arms'][k+k+'n'] and torch.equal(charges[k],w.charge)
                item['self_replay']=check_replay
                check_replay=False
            else:
                for k in MODES:
                    rec=item['arms'][k+k+'n']
                    for m in MODES:
                        item['arms'][k+m+'n']=rec
                        for loc in MODES:item['arms'][k+loc+m]=rec
            for k,w in writers.items():
                w.charge=charges[k]
                w.forced=None
            log.write(json.dumps(item,allow_nan=False)+'\n')
    assert i+1==len(rows) and labels==sum(bool(r['supervised']) for r in rows)
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=len(rows),labels=labels,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('group',choices=('original','additional'))
    for key in ('long-base','plan','quota','drive','fit','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)
