"""Full-validation frozen readout swap with exactly matched predicted Z states."""
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_z_only import ZOnlyUpdater,ZOnlyDetector,make_reader
from opencood.tools.train_int_z_only import engineering
from opencood.tools.train_int_recurrent_deep import evaluate
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


class SameZStateAudit:
    """The evaluator rotates mode order; retain only the first state of each pair."""
    def __init__(self):
        self.pending=None;self.pairs=0

    def __call__(self,name,ego,previous,meta,pred,state,info,index):
        if name not in ('z_actual','z_swapped'):return
        if self.pending is None:
            self.pending=(index,name,state['z']['value'],info['z_refresh'])
        else:
            prior,first,value,refresh=self.pending
            assert prior==index and first!=name,'Audit modes must pair on the same scan'
            assert refresh==info['z_refresh'] and torch.equal(value,state['z']['value']),'Reader swap changed Z trajectory'
            self.pending=None;self.pairs+=1


def audit_readers(a,cfg,train,val,baseline,out,costs,spatial_macs,source):
    path=Path(a.z_only_reader_audit);saved=torch.load(str(path),map_location='cpu');contract=saved['contract']
    assert contract['kind']=='z_only_spatial_v1' and contract['reader']=='joint' and contract['period']==3
    assert contract['source']=={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    original=json.loads((path.parent/'report.json').read_text())
    assert original['status']=='COMPLETED' and original['scans']==5794 and original['labels']==1738
    updater=ZOnlyUpdater(contract['width'],contract.get('context','plain'),contract.get('gate_mode','spatial')).cuda().eval()
    updater.load_state_dict(saved['updater'],strict=True)
    adapted=make_reader(baseline.readout,False).cuda();adapted.load_state_dict(saved['reader'],strict=True)
    fixed=make_reader(baseline.readout,False).cuda()
    actual=ZOnlyDetector(baseline,updater,adapted,3);swapped=ZOnlyDetector(baseline,updater,fixed,3)
    full=ZOnlyDetector(baseline,updater,fixed,1);full_adapted=ZOnlyDetector(baseline,updater,adapted,1)
    write_json(out/'reader_audit_engineering.json',engineering(actual,full,cfg,train))
    swapped.execution_costs=actual.execution_costs;full_adapted.execution_costs=full.execution_costs
    for p in updater.parameters():p.requires_grad_(False)
    write_json(out/'reader_audit_contract.json',dict(checkpoint=str(path),sha256=digest(path),epoch=saved['epoch'],
        source_validation=str(path.parent/'report.json'),source_validation_sha256=digest(path.parent/'report.json'),
        source_contract=contract,split='Full validation5794 scans/1738 labels; no fitting or selection',
        comparison='Same frozen updater and per-scan identical Z; adapted versus fixed mature reader. Independent full-Z adapted/fixed controls.',
        limitations='Reader swap is an off-training-distribution intervention. No training-parameter matched superiority or latency claim; audit timing suppressed.'))
    directory=out/'validation';directory.mkdir()
    tracker=SameZStateAudit()
    report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,saved['epoch'],
        variants=dict(z_actual=actual,z_swapped=swapped,z_full_fixed=full,z_full_adapted=full_adapted),
        expected_ap={'z_actual':original['results']['z_only']['ap70'],
                     'z_full_fixed':.7567910702899006,'z_full_adapted':original['results']['z_full_adapted']['ap70']},
        audit=tracker,share_gpu=True,note='Frozen readout exchange on identical predicted Z; no AP threshold tuning. Audit instrumentation: no timing claim.')
    assert tracker.pending is None and tracker.pairs==5794
    report.update(same_z_scans=tracker.pairs,timing_valid=False)
    for name,result in report['results'].items():
        result['mean_ms']=None;result['p95_ms']=None
        if name!='baseline':
            result['z_full_frames']=result['deep_full_frames'];result['deep_full_frames']=None
            result['reader_dense_frames']=result['readout_full_frames'];result['readout_full_frames']=None
    report['same_z_adapted_minus_fixed_ap70']=report['results']['z_actual']['ap70']-report['results']['z_swapped']['ap70']
    report['full_z_adapted_minus_fixed_ap70']=report['results']['z_full_adapted']['ap70']-report['results']['z_full_fixed']['ap70']
    write_json(directory/'report.json',report);write_json(out/'report.json',report)
    write_json(out/'status.json',dict(status='COMPLETED',scans=5794,same_z_scans=tracker.pairs))
