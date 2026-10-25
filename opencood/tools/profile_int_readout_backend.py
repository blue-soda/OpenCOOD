"""Same-state, same-mask backend comparison. No independent-trajectory AP."""
import copy
import json
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_tiled_readout import TiledReadoutCache
from opencood.tools.int_selection_audit import capture_step
from opencood.tools.run_int_ego import loader,to_device,write_json


def profile_readout_backend(a,cfg,train,modes,out,expected,readout_macs):
    runners=dict(row=modes['selected'],masked_dense=modes['masked_dense'])
    for tile in (8,16):
        runner=copy.copy(modes['selected'])
        runner.readout_cache=TiledReadoutCache(runner.readout,runner.bounds,.5,tile=tile)
        runners['tile{}'.format(tile)]=runner
    contract=json.loads((out/'contract.json').read_text())
    contract.update(run_kind='readout_backend_training_profile',backend_frames=a.readout_backend_frames,
        backend_scope='Same canonical row previous state and input for every backend; only canonical row new state retained. Same final mask and thresholds. Tiles batch ordinary convolutions on core tiles+radius2 inputs, radius1 intermediate; outside-world projected intermediate zeroed. Only original selected cells written. All padded/duplicate tile arithmetic counted. First16 scans warmup. Rotating same-GPU wall model timing includes geometry/index/scatter; no hooks. No AP or independent tiled recurrence claim.')
    write_json(out/'contract.json',contract)
    records={k:[] for k in runners};state=None;maximum=0.
    _,stream=loader(train[:a.readout_backend_frames],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'))
            names=list(runners);offset=index%len(names);names=names[offset:]+names[:offset]
            outputs={};new_states={};masks={}
            for name in names:
                runner=runners[name]
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                pred,new,info,hmask=capture_step(runner,ego,state,sample['meta'])
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                peak=max(0,torch.cuda.max_memory_allocated()-before)
                assert all(torch.isfinite(v).all() for v in pred.values())
                outputs[name]=pred;new_states[name]=new
                masks[name]=(hmask,runner.readout_cache.last_mask)
                total=expected('masked_dense' if name=='masked_dense' else 'selected',info)
                readout_cost=readout_macs if info['readout_refresh'] or name=='masked_dense' else info['readout_extra_linear_macs']
                if index>=16:
                    records[name].append(dict(scan=index+1,refresh=info['readout_refresh'],mean_ms=elapsed,
                        total_gflops=2*total/1e9,readout_gflops=2*readout_cost/1e9,
                        write_fraction=info['readout_write_fraction'],executed_fraction=info['readout_executed_fraction'],
                        peak_bytes=peak))
            for name in runners:
                assert all(torch.equal(x,y) for x,y in zip(masks['row'],masks[name])),'Same-input mask changed'
                for key in outputs['row']:
                    maximum=max(maximum,float((outputs['row'][key]-outputs[name][key]).abs().max()))
                    assert torch.allclose(outputs['row'][key],outputs[name][key],atol=1e-3,rtol=1e-3),(name,key,maximum)
                assert torch.allclose(new_states['row']['readout']['value'],new_states[name]['readout']['value'],atol=1e-3,rtol=1e-3)
                assert torch.equal(new_states['row']['readout']['reference'],new_states[name]['readout']['reference'])
            state=new_states['row']
            if (index+1)%100==0:write_json(out/'status.json',dict(status='RUNNING',phase='readout_backend_profile',scans=index+1))
    assert index+1==a.readout_backend_frames
    summaries={}
    for name,rows in records.items():
        groups={}
        for group in ('all','refresh','partial'):
            selected=[r for r in rows if group=='all' or r['refresh']==(group=='refresh')]
            if not selected:continue
            groups[group]=dict(frames=len(selected),p95_ms=float(np.percentile([r['mean_ms'] for r in selected],95)),
                **{key:float(np.mean([r[key] for r in selected])) for key in
                   ('mean_ms','total_gflops','readout_gflops','write_fraction','executed_fraction')},
                max_incremental_cuda_peak_bytes=max(r['peak_bytes'] for r in selected))
        summaries[name]=dict(groups=groups,readout_packed_weight_bytes=runners[name].readout_cache.packed_weight_bytes)
    write_json(out/'backend_profile.json',dict(status='COMPLETED',split='train',scans=index+1,warmup=16,
        same_input_prediction_max_error=maximum,same_masks_all_frames=True,summaries=summaries,frames=records,
        note='Canonical row state, not independent tiled recurrence; train prefix not fullval AP. Native tile extra padding/halo MACs explicit; no dense-compute-then-mask shortcut.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='readout_backend_profile',scans=index+1))
