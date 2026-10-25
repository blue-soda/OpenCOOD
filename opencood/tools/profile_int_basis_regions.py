"""Bounded real-stream diagnosis; timings here are not full-validation results."""
import functools
import time
from collections import defaultdict
import numpy as np
import torch
from opencood.models.sub_modules import int_basis_spatial_cache as cache_module
from opencood.models.sub_modules import int_basis_spatial_state as stage_module
from opencood.models.sub_modules import int_fused_basis as fused_module
from opencood.models.sub_modules import int_row_spatial_state as row_module
from opencood.models.sub_modules import int_planned_row_state as planned_module
from opencood.tools.run_int_ego import loader, to_device, write_json


class ModuleMacCounter:
    """Executed conv MACs by unique leaf module; excludes nonconv arithmetic."""
    def __init__(self, roots):
        self.hooks=[];self.macs=defaultdict(int);seen=set()
        for prefix,root in roots.items():
            for name,module in root.named_modules():
                if id(module) in seen or not isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                    continue
                seen.add(id(module));label=prefix+'.'+name
                def count(layer,inputs,output,label=label):
                    kh,kw=layer.kernel_size
                    if isinstance(layer,torch.nn.ConvTranspose2d):
                        n,c,h,w=inputs[0].shape
                        value=n*h*w*c*(layer.out_channels//layer.groups)*kh*kw
                    else:value=output.numel()*(layer.in_channels//layer.groups)*kh*kw
                    self.macs[label]+=int(value)
                self.hooks.append(module.register_forward_hook(count))

    def close(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[]


class EventScopes:
    """Nested CUDA event spans plus host enqueue time; never sum nested spans."""
    def __init__(self):
        self.restore = []
        self.events = []

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)
        # Readout current/history are registered submodules, not methods.
        # Instrument their forward without replacing the module/parameters.
        if isinstance(original, torch.nn.Module):
            owner, attribute = original, 'forward'
            original = owner.forward
        @functools.wraps(original)
        def measured(*args, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(); tick = time.perf_counter()
            result = original(*args, **kwargs)
            host = 1000*(time.perf_counter()-tick); end.record()
            self.events.append((label, begin, end, host))
            return result
        setattr(owner, attribute, measured)
        self.restore.append((owner, attribute, original))

    def close(self):
        for owner, attribute, original in reversed(self.restore):
            setattr(owner, attribute, original)
        self.restore = []

    def values(self):
        totals = defaultdict(lambda: [0., 0., 0])
        for label, begin, end, host in self.events:
            totals[label][0] += begin.elapsed_time(end)
            totals[label][1] += host
            totals[label][2] += 1
        return dict(totals)


def profile_stream(a, cfg, val, mode, base, out, parent=None):
    """Normal recurrence is authoritative; replay identical incoming state once.

    All hooks are installed only during the replay and restored even on error.
    Warmup uses the same causal stream. Full refresh/partial frames are separate.
    """
    row_mode=getattr(a,'row_state',False)
    assert (a.fused or row_mode) and a.profile_frames >= 64
    planned=getattr(a,'row_plan',False)
    candidate='planned_row' if planned else ('row' if row_mode else 'fused')
    runners={'base':base,candidate:mode}
    if row_mode:
        assert parent is not None
        runners['original_row' if planned else 'canvas']=parent
    frames = min(a.profile_frames, len(val))
    _, stream = loader(val[:frames], cfg, 2, 303)
    states={name:None for name in runners}
    records = []; maximum = 0.
    with torch.no_grad():
        for index, sample in enumerate(stream):
            ego = to_device(sample['ego'], torch.device('cuda'))
            previous = states[candidate]
            # Alternate ordinary timings to avoid always giving one mode the
            # first-use position. Neither ordinary measurement has event hooks.
            modes=list(runners);offset=index%len(modes);modes=modes[offset:]+modes[:offset]
            times = {}
            for name in modes:
                torch.cuda.synchronize(); tick = time.perf_counter()
                prediction,states[name],detail=runners[name].step(ego,states[name],sample['meta'])
                if name==candidate:pred,state,info=prediction,states[name],detail
                torch.cuda.synchronize(); times[name] = 1000*(time.perf_counter()-tick)
            if index < 16:
                continue
            scopes = EventScopes(); stage = mode.spatial.stage
            counter=ModuleMacCounter(dict(model=mode.model,readout=mode.readout,repair=mode.repair))
            try:
                for owner, attr, label in (
                        (mode.model.pillar_vfe,'forward','pillar_vfe'),
                        (mode.model.scatter,'forward','pillar_scatter'),
                        (mode.model.feature_memory, 'step', 'fm_total'),
                        (mode.spatial, 'step', 'spatial_total'),
                        (stage, 'observe', 'spatial_stem'),
                        (stage, 'dense_from_stem', 'four_layers_dense'),
                        (stage, 'selected_from_stem', 'four_layers_selected'),
                        (mode.spatial, 'warp', 'spatial_warp'),
                        (mode, 'warp', 'deep_warp'),
                        (cache_module, 'early_descriptor', 'descriptor'),
                        (cache_module, 'local_change_mask', 'rms_mask_before_guard_dilation'),
                        (fused_module, 'selected_conv3x3_rows', 'gather_and_selected_linear'),
                        (stage_module, '_write_rows', 'activation_scatter'),
                        (stage_module, '_selected_bn', 'selected_bn'),
                        (mode.repair, 'forward', 'deep_repair'),
                        (mode.readout, 'current', 'readout_current'),
                        (mode.readout, 'history', 'readout_history'),
                        (mode.readout, 'spatial', 'readout_first_spatial'),
                        (mode.readout, 'tail', 'readout_tail'),
                        (mode.readout, 'fuse', 'readout_fuse')):
                    scopes.wrap(owner, attr, label)
                if row_mode:
                    for owner,attr,label in (
                            (row_module,'row_conv','row_gather_conv'),
                            (row_module,'row_project','row_projection'),
                            (row_module,'row_lookup','row_lookup_nested'),
                            (row_module,'_write_rows','final_H_scatter'),
                            (row_module,'_selected_bn','row_bn'),
                            (mode.model.backbone.resnet.layer1,'forward','deep_layer1'),
                            (mode.model.backbone.resnet.layer2,'forward','deep_layer2'),
                            (mode.model.backbone.deblocks[0],'forward','deblock_current'),
                            (mode.model.backbone.deblocks[1],'forward','deblock_deep1'),
                            (mode.model.backbone.deblocks[2],'forward','deblock_deep2'),
                            (mode.model.cls_head,'forward','classification_head'),
                            (mode.model.reg_head,'forward','regression_head')):
                        scopes.wrap(owner,attr,label)
                if planned:
                    for attr,label in (('dependency_plan','shared_dependency_plan'),('planned_conv','planned_gather_conv'),
                                       ('row_project','planned_projection'),('_write_rows','planned_final_H_scatter')):
                        scopes.wrap(planned_module,attr,label)
                torch.cuda.synchronize(); tick = time.perf_counter()
                replay, replay_state, replay_info = mode.step(ego, previous, sample['meta'])
                torch.cuda.synchronize(); instrumented_ms = 1000*(time.perf_counter()-tick)
                spans = scopes.values()
            finally:
                scopes.close()
                counter.close()
            assert info['spatial_selected_cells'] == replay_info['spatial_selected_cells']
            for key in pred:
                maximum = max(maximum, float((pred[key]-replay[key]).abs().max()))
                assert torch.equal(pred[key], replay[key]), 'Instrumentation changed prediction'
            for left, right in zip(state['spatial']['maps'], replay_state['spatial']['maps']):
                assert torch.equal(left, right), 'Instrumentation changed cached state'
            macs=dict(counter.macs)
            macs['spatial_update_functional']=int(replay_info['spatial_extra_linear_macs'])
            records.append(dict(index=index, refresh=info['spatial_refresh'], deep_refresh=not info['local_partial'],area=info['spatial_write_fraction'],
                                ordinary_ms=times, instrumented_ms=instrumented_ms, scopes=spans,conv_linear_macs=macs))
            if (index+1) % 64 == 0:
                write_json(out/'status.json', dict(status='RUNNING', phase='profile', scans=index+1))
    groups = {}
    for name, group in (('all', records), ('refresh', [r for r in records if r['refresh']]),
                        ('partial', [r for r in records if not r['refresh']])):
        assert group, 'Profile needs both refresh and partial frames'
        labels = sorted(set(k for row in group for k in row['scopes']))
        groups[name] = dict(frames=len(group), area=float(np.mean([r['area'] for r in group])),
            ordinary_mean_ms={k:float(np.mean([r['ordinary_ms'][k] for r in group])) for k in runners},
            instrumented_mean_ms=float(np.mean([r['instrumented_ms'] for r in group])),
            scopes={k:dict(cuda_span_ms_per_frame=float(np.mean([r['scopes'].get(k, [0,0,0])[0] for r in group])),
                           host_ms_per_frame=float(np.mean([r['scopes'].get(k, [0,0,0])[1] for r in group])),
                           calls_per_frame=float(np.mean([r['scopes'].get(k, [0,0,0])[2] for r in group]))) for k in labels})
        mac_labels=sorted(set(k for row in group for k in row['conv_linear_macs']))
        groups[name]['conv_linear_gflops_by_module']={k:2.*sum(r['conv_linear_macs'].get(k,0) for r in group)/len(group)/1e9 for k in mac_labels}
        groups[name]['conv_linear_gflops']=sum(groups[name]['conv_linear_gflops_by_module'].values())
        groups[name]['deep_refresh_frames']=sum(r['deep_refresh'] for r in group)
    write_json(out/'profile.json', dict(status='COMPLETED', scans=frames, warmup=16, groups=groups,
        modes=list(runners),row_execution=row_mode,planned_row_execution=planned,
        exact_replay_max_error=maximum, frames=records,
        scope='Diagnostic first-N real validation scans, not AP or whole-dataset timing. Independent causal states for each named mode; instrumented candidate replay uses identical incoming state and leaves authoritative normal recurrence intact. CUDA events and host spans are nested; do not sum them. CUDA spans can include host-launch gaps and MAC-hook overhead. Conv attribution counts unique Conv/ConvTranspose leaves plus explicit selected spatial linear MACs; excludes other arithmetic, Pillar linear and memory movement. Module costs add without overlap; timing spans do not. No internal synchronizations added; event setup itself has overhead. Ordinary same-GPU rotating timings exclude hooks/IO/H2D/postprocess.'))
    write_json(out/'status.json', dict(status='COMPLETED', phase='profile'))
