"""Same-state bounded row batching comparison, with separate intrusive timings."""
import copy
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_local_current_readout import LocalCurrentReadoutCache
import opencood.models.sub_modules.int_local_current_readout as cache_module
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.train_int_recurrent_deep import cost_of
from opencood.tools.int_gpu_exclusivity import sample_gpu_usage


def profile(a,cfg,train,canonical,out,costs,spatial_macs):
    runners=dict(row32768=canonical)
    for name,execution,chunk in (('row4096','selected',4096),('masked_dense','masked_dense',32768),('cached_geometry','selected',32768)):
        mode=copy.copy(canonical)
        mode.readout_cache=LocalCurrentReadoutCache(mode.readout,canonical.readout_cache.repair,mode.bounds,canonical.readout_cache.threshold,execution,3,row_chunk_size=chunk,cache_geometry=name=='cached_geometry')
        runners[name]=mode
    write_json(out/'current_backend_contract.json',dict(split='train',frames=512,warmup=16,
        scope='Same canonical previous state/input/final masks. Only row32768 state advances. No independent AP or new selection.',
        model_timing='Uninstrumented synchronized model wall time; rotate/reverse order; peak incremental allocated memory. MAC engineering done in separate first8 replays.',
        phase_timing='Additional readout-only replay every8 frames. Synchronized warp/selection/compute boundaries and nested geometry, gathered first/second convolutions, scatter. Nested times are INCLUDED in compute, not additive to it. Refresh remainder includes mature convolutions. Intrusive times alter scheduling and are NOT model timing.',
        limits='Chunk4096 versus32768, dense masked control and geometry-only fixed-grid lookup with chunk32768. No tile/threshold search. Geometry construction occurs on first use; only steady-state timing after warmup. Geometry memory explicit. Readout-only replay and all discarded states excluded from model timing.'))
    records={k:[] for k in runners};phases={k:[] for k in runners};state=None;maximum=0.;isolation=[]
    def guard(scan):
        try:evidence=sample_gpu_usage(a.gpu)
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
                    counter=ConvCounter(mode.model,torch.nn.ModuleList([mode.readout,mode.repair,mode.readout_cache.repair]))
                    try:_,_,measured=mode.step(ego,previous,meta)
                    finally:counter.close()
                    actual=counter.total+measured['spatial_extra_linear_macs']+measured.get('readout_extra_linear_macs',0)
                    assert actual==cost_of(costs,spatial_macs,measured,True)
            for name in runners:
                assert torch.equal(masks['row32768'],masks[name])
                assert torch.equal(states['row32768']['readout']['reference'],states[name]['readout']['reference'])
                assert torch.equal(states['row32768']['deep']['value'],states[name]['deep']['value'])
                for k in predictions[name]:
                    maximum=max(maximum,float((predictions[name][k]-predictions['row32768'][k]).abs().max()))
                    assert torch.allclose(predictions[name][k],predictions['row32768'][k],atol=1e-3,rtol=1e-3)
                assert torch.allclose(states[name]['readout']['value'],states['row32768']['readout']['value'],atol=1e-3,rtol=1e-3)
            for name in ('row4096','cached_geometry'):
                assert cost_of(costs,spatial_macs,infos[name],True)==cost_of(costs,spatial_macs,infos['row32768'],True)
            state=states['row32768']
            if index>=16 and index%8==0:
                # Capture the actual readout input in a discarded canonical replay.
                cache=canonical.readout_cache;original=cache.step;captured={}
                def capture(current,old,metadata,reset=False):
                    captured.update(current=current,state=old,meta=metadata,reset=reset)
                    return original(current,old,metadata,reset)
                cache.step=capture
                try:other,_,_=canonical.step(ego,previous,meta)
                finally:cache.step=original
                for k in other:assert torch.equal(other[k],predictions['row32768'][k])
                for name in names:
                    cache=runners[name].readout_cache;timings={};originals={k:getattr(cache,k) for k in ('select_mask','selected','dense','neighborhood_plan')}
                    module_originals={k:getattr(cache_module,k) for k in ('warp_bev','row_lookup','prepared_conv3x3_rows','planned_conv','_write_rows')}
                    def timed(key,fn):
                        def call(*args,**kwargs):
                            torch.cuda.synchronize();tick=time.perf_counter();result=fn(*args,**kwargs);torch.cuda.synchronize()
                            timings[key]=timings.get(key,0.)+(time.perf_counter()-tick)*1000;return result
                        return call
                    for key,fn in originals.items():setattr(cache,key,timed(key,fn))
                    for key,fn in module_originals.items():setattr(cache_module,key,timed(key,fn))
                    try:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        output,_,info=cache.step(captured['current'],captured['state'],captured['meta'],captured['reset'])
                        torch.cuda.synchronize();elapsed=(time.perf_counter()-tick)*1000
                    finally:
                        for key,fn in originals.items():setattr(cache,key,fn)
                        for key,fn in module_originals.items():setattr(cache_module,key,fn)
                    assert torch.allclose(output,states[name]['readout']['value'],atol=1e-3,rtol=1e-3)
                    compute=timings.get('selected',timings.get('dense',0.))
                    selection=timings.get('select_mask',0.);warp=timings.get('warp_bev',0.)
                    phases[name].append(dict(scan=index+1,refresh=info['readout_refresh'],total_ms=elapsed,
                        selection_ms=selection,compute_ms=compute,warp_ms=warp,
                        geometry_ms=timings.get('neighborhood_plan',0.)+timings.get('row_lookup',0.),
                        first_rows_ms=timings.get('prepared_conv3x3_rows',0.),second_rows_ms=timings.get('planned_conv',0.),
                        scatter_ms=timings.get('_write_rows',0.),remainder_ms=elapsed-selection-compute-warp))
            if (index+1)%100==0:
                guard(index+1);write_json(out/'status.json',dict(status='RUNNING',phase='current_backend_profile',scans=index+1))
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
                intrusive_phase_samples=len(timing),intrusive_phases={k:float(np.mean([r[k] for r in timing])) if timing and valid else None for k in ('total_ms','selection_ms','compute_ms','warp_ms','geometry_ms','first_rows_ms','second_rows_ms','scatter_ms','remainder_ms')})
        summaries[name]=groups
    write_json(out/'current_backend_profile.json',dict(status='COMPLETED',split='train',scans=512,warmup=16,timing_valid=valid,
        geometry_bytes={name:runner.readout_cache.geometry_bytes for name,runner in runners.items()},
        same_input_prediction_max_error=maximum,actual_macs_verified=True,summaries=summaries,frames=records,phases=phases,
        note='Canonical row32768 trajectory only. No AP result. Intrusive stage times are diagnostic, distinct from uninstrumented model measurements.'))
    write_json(out/'status.json',dict(status='COMPLETED',phase='current_backend_profile',scans=512))
