"""Same-GPU fixed-checkpoint comparison of complete-D recurrence periods."""
import copy
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_recurrent_deep import RecurrentDeepDetector
from opencood.tools.int_artifact_paths import artifact_path
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of,evaluate


def compare_frozen(a,cfg,train,val,baseline,out,costs,spatial_macs):
    source=json.loads((out/'contract.json').read_text())
    required={k:source[k] for k in ('source','memory','readout','repair','basis','data')}
    repairs={};checkpoints={};expected={}
    for period,path in ((3,a.recurrent_three_checkpoint),(6,a.recurrent_six_checkpoint)):
        path=Path(artifact_path(path));saved=torch.load(str(path),map_location='cpu')
        assert saved['contract']['period']==period and saved['contract']['source']==required
        report=json.loads((path.parent/'report.json').read_text())
        assert report['status']=='COMPLETED' and report['scans']==5794 and report['labels']==1738
        repair=copy.deepcopy(baseline.repair).eval();repair.load_state_dict(saved['repair'],strict=True)
        for p in repair.parameters():p.requires_grad_(False)
        repairs[period]=repair;expected['trained%d_period%d'%(period,period)]=report['results']['recurrent']['ap70']
        checkpoints[period]=dict(path=str(path),sha256=digest(path),epoch=saved['epoch'],trained_period=period,expected_ap70=expected['trained%d_period%d'%(period,period)])
    modes={name:RecurrentDeepDetector(baseline,repairs[trained],period) for name,trained,period in
        [('trained3_period3',3,3),('trained6_period3',6,3),('trained6_period4',6,4),('trained6_period6',6,6)]}
    contract=dict(checkpoints=checkpoints,period_age_us={name:mode.deep.max_age_us for name,mode in modes.items()},
        scope='Frozen comparison only. Preserve four-layer H baseline and dense readout; ordinary D updater dense. No optimizer or local D skipping.',
        attribution='Period3-trained control plus SAME period6-trained weights evaluated at3/4/6. Period4 is deployment transfer without period4 training; no causal claim about training length.',
        validation='5794 scans/1738 labels, independent complete histories, all five modes rotate/reverse on SAME GPU. Baseline and matching-period checkpoint AP must reproduce.',
        selection='Use measured accuracy/actual conv-linear FLOPs/model time to choose refresh period before local update; training may continue independently on a separate GPU.')
    write_json(out/'frozen_recurrent_contract.json',contract)
    states={name:None for name in modes};partial={name:0 for name in modes}
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'))
            for name,mode in modes.items():
                counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
                try:pred,states[name],info=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                assert counter.total+info['spatial_extra_linear_macs']==cost_of(costs,spatial_macs,info,True)
                assert all(torch.isfinite(x).all() for x in pred.values());partial[name]+=int(info['deep_partial'])
    assert all(partial.values())
    write_json(out/'frozen_recurrent_engineering.json',dict(status='PASSED',frames=32,partial_frames=partial,actual_macs_verified=True))
    directory=out/'full';directory.mkdir()
    report=evaluate(a,baseline,None,cfg,val,directory,costs,spatial_macs,'frozen',variants=modes,expected_ap=expected)
    report['frozen_contract']=contract
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED',scans=len(val)))
