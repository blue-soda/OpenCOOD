"""Same-state bounded row batching comparison, with separate intrusive timings."""
import copy
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_local_readout import LocalReadoutCache
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of
from opencood.tools.int_gpu_exclusivity import check_gpu_exclusive


def profile_composition(a,cfg,train,canonical,out,costs,spatial_macs):
    runners=dict(row4096=canonical)
    for name,execution,chunk in (('row32768','selected',32768),('masked_dense','masked_dense',4096)):
        mode=copy.copy(canonical)
        mode.readout_cache=LocalReadoutCache(mode.readout,mode.bounds,canonical.readout_cache.threshold,execution,3,row_chunk_size=chunk)
        runners[name]=mode
    write_json(out/'recurrent_backend_contract.json',dict(split='train',frames=512,warmup=16,
        scope='Same canonical previous state/input/final masks. Only row4096 state advances. No independent AP or new selection.',
        model_timing='Uninstrumented synchronized model wall time; rotate/reverse order; peak incremental allocated memory. MAC engineering done in separate first8 replays.',
        phase_timing='Additional readout-only replay every8 frames, synchronization at select_mask and selected/dense boundaries. These intrusive phase times are NOT model timing and may alter scheduling. Remainder includes warp, packing, reference merge, metadata and synchronization.',
        limits='Only chunk4096 versus32768 plus dense masked control. No tile/threshold search. Readout-only replay and all discarded states excluded from model timing.'))
    records={k:[] for k in runners};phases={k:[] for k in runners};state=None;maximum=0.;isolation=[]
    def guard(scan):
        try:evidence=check_gpu_exclusive(a.gpu)
        except Exception as exc:
            write_json(out/'backend_gpu_exclusivity.json',dict(status='INVALID',scan=scan,error=repr(exc),evidence=getattr(exc,'evidence',[])));raise
        evidence['scan']=scan;isolation.append(evidence)
        write_json(out/'backend_gpu_exclusivity.json',dict(status='SAMPLED_CLEAR' if all(r['timing_valid'] for r in isolation) else 'INVALID_TIMING',samples=isolation))
    guard(0)
    _,stream=loader(train[:512],cfg,0,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];previous=state
            names=list(runners);offset=index%len(names);names=names[offset:]+names[:offset]
            if (index//len(names))%2:names.reverse()
            predictions={};states={};masks={};infos={}
            for name in names:
                mode=runners[name]
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                pred,new,info=mode.step(ego,previous,meta)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000;peak=max(0,torch.cuda.max_memory_allocated()-before)
                predictions[name]=pred;states[name]=new;infos[name]=info;masks[name]=mode.readout_cache.last_mask.clone()
                if index>=16:records[name].append(dict(scan=index+1,refresh=info['readout_refresh'],mean_ms=elapsed,
                    gflops=2*cost_of(costs,spatial_macs,info,True)/1e9,peak_bytes=peak,
                    executed_fraction=info['readout_executed_fraction'],write_fraction=info['readout_write_fraction']))
                if index<8:
                    counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair]))
                    try:_,_,measured=mode.step(ego,previous,meta)
                    finally:counter.close()
                    actual=counter.total+measured['spatial_extra_linear_macs']+measured.get('readout_extra_linear_macs',0)
                    assert actual==cost_of(costs,spatial_macs,measured,True)
            for name in runners:
                assert torch.equal(masks['row4096'],masks[name])
                assert torch.equal(states['row4096']['readout']['reference'],states[name]['readout']['reference'])
                assert torch.equal(states['row4096']['deep']['value'],states[name]['deep']['value'])
                for k in predictions[name]:
                    maximum=max(maximum,float((predictions[name][k]-predictions['row4096'][k]).abs().max()))
                    assert torch.allclose(predictions[name][k],predictions['row4096'][k],atol=1e-3,rtol=1e-3)
                assert torch.allclose(states[name]['readout']['value'],states['row4096']['readout']['value'],atol=1e-3,rtol=1e-3)
            assert cost_of(costs,spatial_macs,infos['row4096'],True)==cost_of(costs,spatial_macs,infos['row32768'],True)
            state=states['row4096']
            if index>=16 and index%8==0:
                # Capture the actual readout input in a discarded canonical replay.
                cache=canonical.readout_cache;original=cache.step;captured={}
                def capture(current,old,metadata,reset=False):
                    captured.update(current=current,state=old,meta=metadata,reset=reset)
                    return original(current,old,metadata,reset)
                cache.step=capture
                try:other,_,_=canonical.step(ego,previous,meta)
                finally:cache.step=original
                for k in other:assert torch.equal(other[k],predictions['row4096'][k])
                for name in names:
                    cache=runners[name].readout_cache;timings={};originals={k:getattr(cache,k) for k in ('select_mask','selected','dense')}
                    def timed(key,fn):
                        def call(*args,**kwargs):
                            torch.cuda.synchronize();tick=time.perf_counter();result=fn(*args,**kwargs);torch.cuda.synchronize()
                            timings[key]=timings.get(key,0.)+(time.perf_counter()-tick)*1000;return result
                        return call
                    for key,fn in originals.items():setattr(cache,key,timed(key,fn))
                    try:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        output,_,info=cache.step(captured['current'],captured['state'],captured['meta'],captured['reset'])
                        torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000
                    finally:
                        for key,fn in originals.items():setattr(cache,key,fn)
                    assert torch.allclose(output,states[name]['readout']['value'],atol=1e-3,rtol=1e-3)
                    phases[name].append(dict(scan=index+1,refresh=info['readout_refresh'],total_ms=elapsed,
                        selection_ms=timings.get('select_mask',0.),compute_ms=timings.get('selected',0.)+timings.get('dense',0.),
                        remainder_ms=elapsed-sum(timings.values())))
            if (index+1)%100==0:
                guard(index+1);write_json(out/'status.json',dict(status='RUNNING',phase='recurrent_backend_profile',scans=index+1))
    guard(512);valid=all(r['timing_valid'] for r in isolation);summaries={}
    for name in runners:
        groups={}
        for group in ('all','refresh','partial'):
            rows=[r for r in records[name] if group=='all' or r['refresh']==(group=='refresh')]
            timing=[r for r in phases[name] if group=='all' or r['refresh']==(group=='refresh')]
            if not rows:continue
            groups[group]=dict(frames=len(rows),gflops=float(np.mean([r['gflops'] for r in rows])),
                mean_ms=float(np.mean([r['mean_ms'] for r in rows])) if valid else None,
                p95_ms=float(np.percentile([r['mean_ms'] for r in rows],95)) if valid else None,
                max_incremental_cuda_peak_bytes=max(r['peak_bytes'] for r in rows),
                write_fraction=float(np.mean([r['write_fraction'] for r in rows])),
                executed_fraction=float(np.mean([r['executed_fraction'] for r in rows])),
                intrusive_phase_samples=len(timing),intrusive_phases={k:float(np.mean([r[k] for r in timing])) if timing and valid else None for k in ('total_ms','selection_ms','compute_ms','remainder_ms')})
        summaries[name]=groups
    write_json(out/'recurrent_backend_profile.json',dict(status='COMPLETED',split='train',scans=512,warmup=16,timing_valid=valid,
        same_input_prediction_max_error=maximum,actual_macs_verified=True,summaries=summaries,frames=records,phases=phases,
        note='Canonical row4096 trajectory only. No AP result. Intrusive stage times are diagnostic, distinct from uninstrumented model measurements.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='recurrent_backend_profile',scans=512))
