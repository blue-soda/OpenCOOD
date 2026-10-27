"""Real causal spatial activation reuse with explicit executed-area/MAC costs."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_basis_spatial_state import SpatialBasisStage
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_dependency_spatial_state import DependencyProjectedSpatialStage
from opencood.models.sub_modules.int_single_spatial_state import SingleSpatialStateStage
from opencood.models.sub_modules.int_row_spatial_state import RowSpatialStateStage
from opencood.models.sub_modules.int_planned_row_state import PlannedRowSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCacheDetector
from opencood.models.sub_modules.int_fused_basis import fuse_spatial_basis
from opencood.models.sub_modules.int_cached_metric_warp import CachedMetricWarp
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout, SplitReadoutDetector
from opencood.models.sub_modules.int_light_repair import LightRepair
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path,relocate_tree,read_mapping
from opencood.utils import eval_utils


def worker(a):
    out = Path(a.output); write_json(out/'status.json', dict(status='RUNNING', phase='setup'))
    if a.path_map:
        os.environ['OPENCOOD_INT_PATH_MAP']=str(Path(a.path_map).resolve())
        write_json(out/'path_relocation.json',dict(mapping=dict(read_mapping(os.environ['OPENCOOD_INT_PATH_MAP'])),sha256=digest(a.path_map),
            policy='Only runtime file locations change; source bytes, SHA identities, contracts and resume optimizer remain unchanged.'))
    for k, v in dict(mode='gru', stage='pilot', device='cuda', content_calibration=None, periodic_last=0, max_age=.25).items():
        setattr(a, k, v)
    base, cfg, val, source = setup(a)
    rb, readout = completed_best(a.readout_run)
    artifacts = []
    spatial_run = a.projected_run or a.basis_run
    for root, sha in ((a.memory_run, '1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'),
                      (a.repair_run, '9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'),
                      (spatial_run, None if a.projected_run else 'c2e173eeed6dd4019319d1f421f81b2582fcf9687184a98a3091c36b16f3c928')):
        root = Path(root)
        assert json.loads((root/'status.json').read_text())['status'] == 'COMPLETED'
        best = json.loads((root/'best.json').read_text())
        assert digest(artifact_path(best['checkpoint'])) == best['sha256']
        if sha is not None: assert best['sha256'] == sha
        artifacts.append((best, torch.load(artifact_path(best['checkpoint']), map_location='cpu')))
    (mb, memory), (pb, repaired), (sb, spatial) = artifacts
    assert memory['contract']['data'] == repaired['contract']['data'] == spatial['contract']['data'] == source['contract']
    assert memory['contract']['source']['sha256'] == readout['contract']['source_sha256'] == source['checkpoint']['sha256']
    assert repaired['contract']['readout'] == spatial['contract']['readout'] == rb
    assert spatial['contract']['memory'] == mb and spatial['contract']['repair'] == pb
    if a.projected_run:
        assert spatial['contract']['rank'] in (64,96)
        assert spatial['contract']['phase'] == 'Dense precision recovery for projected four-map spatial update; no new spatial cache during training/validation'
        dense_report = json.loads((Path(artifact_path(sb['checkpoint'])).parent/'report.json').read_text())
        assert spatial['epoch'] == sb['epoch']
    else:
        assert spatial['contract']['rank'] == 9
        dense_report = json.loads((Path(a.basis_run)/'report.json').read_text())
    assert dense_report['status'] == 'COMPLETED'
    for name in ('full', 'cached'):
        if not a.projected_run:
            assert abs(dense_report['results'][name]['ap70']-dense_report['results']['base_'+name]['ap70']) < 1e-5
    assert abs(dense_report['results']['cached']['ap70']-sb['cached_ap70']) < 1e-8
    base.shrink_conv = compact_shrink(base.shrink_conv, 96)[0].cuda()
    base.shrink_conv.load_state_dict(readout['readout'], strict=True)
    base.feature_memory = projected_gru(base.feature_memory, 32)[0].cuda()
    base.feature_memory.load_state_dict(memory['memory'], strict=True); base.eval()
    split = SplitCompactReadout(base.shrink_conv, base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair = LightRepair(96, width=32).cuda().eval(); repair.load_state_dict(repaired['repair'], strict=True)
    student = copy.deepcopy(base)
    stage = (ProjectedSpatialStage(base.backbone.resnet.layer0, spatial['contract']['rank']) if a.projected_run
             else SpatialBasisStage(base.backbone.resnet.layer0, 9)).cuda().eval()
    stage.load_state_dict(spatial['stage'], strict=True)
    parent_stage = stage
    if a.compact_state: stage = CompactProjectedSpatialStage(stage).cuda().eval()
    if a.dependency_masks: stage = DependencyProjectedSpatialStage(stage).cuda().eval()
    if a.single_state:stage=SingleSpatialStateStage(stage).cuda().eval()
    if a.row_state:stage=RowSpatialStateStage(stage).cuda().eval()
    if a.row_plan:stage=PlannedRowSpatialStage(stage).cuda().eval()
    student.backbone.resnet.layer0 = stage
    for m in (base, student, split, repair):
        for p in m.parameters(): p.requires_grad_(False)
    modes = dict(base=SplitReadoutDetector(base, cfg, split, 3, repair),
        dense=SplitReadoutDetector(student, cfg, split, 3, repair),
        selected=SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, confidence_threshold=a.confidence_guard,cell_age_limit_s=.25 if a.cell_age else None,spatial_dilation=0 if a.dependency_masks else 4,confidence_dilation=4 if a.dependency_guard else None),
        masked_dense=SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, execution='masked_dense', confidence_threshold=a.confidence_guard,cell_age_limit_s=.25 if a.cell_age else None,spatial_dilation=0 if a.dependency_masks else 4,confidence_dilation=4 if a.dependency_guard else None))
    roots = {name:base if name == 'base' else student for name in modes}
    if a.compact_state and not a.cell_age:
        if a.dependency_masks:parent_stage=CompactProjectedSpatialStage(parent_stage).cuda().eval()
        if a.single_state:parent_stage=DependencyProjectedSpatialStage(parent_stage).cuda().eval()
        if a.row_state:parent_stage=SingleSpatialStateStage(parent_stage).cuda().eval()
        if a.row_plan:parent_stage=RowSpatialStateStage(parent_stage).cuda().eval()
        parent_model = copy.deepcopy(student); parent_model.backbone.resnet.layer0 = parent_stage
        for p in parent_model.parameters():p.requires_grad_(False)
        roots['parent_selected'] = parent_model
        modes['parent_selected'] = SpatialBasisCacheDetector(parent_model,cfg,split,repair,a.threshold,confidence_threshold=a.confidence_guard,
            spatial_dilation=0 if a.single_state else 4,confidence_dilation=4 if a.single_state else None)
        if a.dependency_masks and not a.single_state:
            roots['uniform_core']=parent_model
            modes['uniform_core']=SpatialBasisCacheDetector(parent_model,cfg,split,repair,a.threshold,confidence_threshold=a.confidence_guard,spatial_dilation=0,confidence_dilation=4 if a.dependency_guard else None)
    if a.cell_age:
        roots['periodic_selected']=student
        modes['periodic_selected']=SpatialBasisCacheDetector(student,cfg,split,repair,a.threshold,confidence_threshold=a.confidence_guard,
            spatial_dilation=0 if a.dependency_masks else 4,confidence_dilation=4 if a.dependency_guard else None)
    if a.fused:
        fused = copy.deepcopy(student); fused.backbone.resnet.layer0 = fuse_spatial_basis(stage)
        roots['fused'] = fused
        modes['fused'] = SpatialBasisCacheDetector(fused, cfg, split, repair, a.threshold, confidence_threshold=a.confidence_guard)
    if a.prepared:
        prepared = copy.deepcopy(student); prepared.backbone.resnet.layer0 = fuse_spatial_basis(stage, prepared=True)
        roots['prepared'] = prepared
        modes['prepared'] = SpatialBasisCacheDetector(prepared, cfg, split, repair, a.threshold, confidence_threshold=a.confidence_guard)
    if a.fast_warp:
        roots['fast_warp'] = prepared
        modes['fast_warp'] = SpatialBasisCacheDetector(prepared, cfg, split, repair, a.threshold,
            confidence_threshold=a.confidence_guard, warp=CachedMetricWarp(cfg['model']['args']['lidar_range']))
    if a.cache_fp16:
        roots['cache_fp16'] = prepared
        modes['cache_fp16'] = SpatialBasisCacheDetector(prepared, cfg, split, repair, a.threshold,
            confidence_threshold=a.confidence_guard, cache_dtype=torch.float16)
    if a.long_history:
        for name, execution in (('long_history','selected'),('long_masked','masked_dense')):
            roots[name] = prepared
            modes[name] = SpatialBasisCacheDetector(prepared,cfg,split,repair,a.threshold,period=5,
                execution=execution,confidence_threshold=a.confidence_guard,cache_dtype=torch.float16,max_full_age_s=.45)
    control = SpatialBasisCacheDetector(student, cfg, split, repair, a.threshold, period=1, confidence_threshold=a.confidence_guard,spatial_dilation=0 if a.dependency_masks else 4)
    write_json(out/'contract.json', dict(source=source['checkpoint'], data=source['contract'], memory=mb, readout=rb,
        repair=pb, basis=sb, threshold=a.threshold, confidence_guard=a.confidence_guard, fused_execution=a.fused,
        projected_spatial=bool(a.projected_run),
        compact_projected_state=a.compact_state,
        cell_age_policy=a.cell_age,
        dependency_masks=a.dependency_masks,
        dependency_guard=a.dependency_guard,
        single_spatial_state=a.single_state,
        row_spatial_execution=a.row_state,
        planned_row_execution=a.row_plan,
        row_plan_contract='When enabled overrides parent: parent_selected is original active-row single-H implementation. Selected prepares all four neighbor/residual index maps once per frame and pre-packs fixed3x3 weights once. Same masks, row arithmetic, MACs and history. Packed weight buffers counted separately from temporal state; no persistent geometry or activation caches.',
        row_contract='If enabled, parent_selected is SAME single-H periodic policy with temporary full canvases. Selected instead carries temporary active rows; four projections execute supports4/3/2/1 and four3x3 execute3/2/1/0. All projection MACs explicit, no dense projection hooks in partial selected path. Same weights, final core, confidence4, period3 and one persistent H; same dense/masked controls. Index lookup, gather, validation and final H clone remain timed. No new training or age policy.',
        single_state_contract='If enabled overrides parent/state descriptions: selected/masked_dense persist ONLY final128-channel full H, with reference/confidence; intermediate96/128/96 activations are ZERO temporary canvases consumed only inside current dependency supports, never historical tensors. Without cell_age, parent_selected retains FOUR maps with SAME dependency3/2/1/0 policy and confidence radius4. With cell_age, periodic_selected instead is SAME single-H period3/.25s control; selected/masked_dense add age>=.25s cells to final core BEFORE complete upstream supports. No old intermediate values contribute to recomputed cells, but time since direct computation does not guarantee accurate stale features under motion. Dense/current weights unchanged; no uniform_core mode. Four-layer update areas differ from one-map persistent-write area. No new parameters/training.',
        dependency_contract='If enabled, overrides generic dilation/compact-parent descriptions: FINAL core unions undilated current content tiles, previous own confidence support, and uncovered cells. Confidence radius is4 if dependency_guard, otherwise0; unchanged threshold.1. Four layer supports are core dilated3/2/1/0, guaranteeing complete CURRENT stem dependencies at updated final cells (fixed BN). Each intermediate writes its own support; descriptor updates only final core. Parent_selected is compact original shared content/confidence radius4; uniform_core uses the same core policy as selected but no upstream dependency halo, an attribution control. All weights/checkpoints/refresh fixed; no training. Layer areas include supports and full refresh. Selected core exactness does not guarantee unchanged cached cells are accurate.',
        cell_age_contract='If enabled: selected/masked_dense store and nearest-warp a FP32 cell-age map; add inter-frame dt and force cells aged>=.25s into mask, reset selected ages0. Same content/own-confidence dilation; expiry union has no extra dilation. Scene/FM reset still full. No periodic/global-full-age refresh; period3/.25s is retained only in periodic_selected control and unchanged deep cache. Per-cell direct age<.25s is NOT a bound on accumulated multi-layer dependency error. No motion prediction, quantization, training or threshold change.',
        compact_state_contract='Optional: cache conv2 input projection(r channels) instead of full conv1 post-ReLU(128), retain complete block outputs128. Same weights/nonlinearities/residuals/nearest geometry. On partial frames conv1 input projection dense, conv2 input projection only selected rows; selected3x3 unchanged. Same compact masked-dense control and original four128-map parent_selected paired. No quantization/training.',
        projected_contract='If enabled: completed recovery run best-by-fullAP checkpoint, own dense baseline replay, dense 1x1 projection every frame and selected 3x3 output rows. All four history maps FP32, spatial/deep period3/.25s. No additional training or fused kernels. Compare selected to SAME compressed checkpoint dense and masked-dense recurrence; base is mature uncompressed spatial path.',
        profile_frames=a.profile_frames, run_kind='selection_divergence_diagnostic' if a.audit_selection else ('bounded_timing_profile' if a.profile_frames else 'full_validation'),
        selection_audit=a.audit_selection,
        verify_recurrence=a.verify_recurrence,
        recurrence_verification='Optional: compare actual own-state masks outside timing; log each divergence and require same-input replay mask equality and output/state closeness. Replays discarded. Capture-mask overhead included in model timing, replay excluded. No threshold rounding or trajectory injections. Maximum128 divergent frames, else fail for review.',
        prepared_execution=a.prepared,
        fast_warp_execution=a.fast_warp,
        cache_fp16=a.cache_fp16,
        long_history=a.long_history,
        long_history_contract='Only experimental spatial-cache policy uses period5 and max full age .45s; paired cache_fp16 remains period3/.25s. FM gap reset and deep anchor period3/.25s unchanged for all modes. Same weights, thresholds, FP16 activation storage and FP32 compute. No teacher state or fine tuning. Full validation base/cache_fp16/long_history; engineering also checks long-history masked-dense recurrence.',
        cache_fp16_contract='Only four post-activation history maps saved as FP16; reference, confidence, FM, deep anchor and all current compute/weights stay FP32. Promote history to FP32 during packing before alignment. No AMP, clipping or training. Rounded history may change future predictions and masks; measure complete AP and actual executed costs. Dtype-conversion time and memory included. Full validation base/prepared/cache_fp16.',
        fast_warp_contract='Static metric cell centers cached; same SE2 matrix/grid arithmetic and nearest sampling. Out-of-coverage coordinates set outside before sampling instead of a channelwise output multiply. FM alignment unchanged. Per-mode persistent static geometry bytes separate from recurrent state. Engineering all modes; full validation base/prepared/fast_warp only if enabled.',
        prepared_contract='Optional shared per-frame mask/nonzero/neighborhood indices for four layers; prepacked immutable weights stored as nonpersistent buffers; linear chunks16384 vs original4096. Identical masks/caches/weights/MACs. Extra packed weight bytes reported separately from recurrent state. No persistent index plan or activation patches.',
        execution_fusion='Optional inference-only composition of full-rank basis+mix into native3x3. Refresh uses native conv; partial gathers selected neighborhoods and computes selected output rows. Replaces basis/mix weights; unchanged four activation caches, reference/guard and mask. No extra dense spatial basis. Count gathered standard convolution MACs, not basis-factor arithmetic.',
        spatial_period=3, max_full_age_s=.25, deep_period=3,
        spatial_policy_by_mode={k:dict(period=m.spatial.period,max_full_age_s=m.spatial.max_full_age_us/1e6,cell_age_limit_s=m.spatial.cell_age_limit_s,global_refresh_enabled=m.spatial.cell_age_limit_s is None,content_dilation=m.spatial.dilation,confidence_dilation=m.spatial.confidence_dilation) for k,m in modes.items() if hasattr(m,'spatial')},
        selector='Current mature stem16-channel mean descriptor at native grid; 4x4-tile RMS relative change to aligned retained reference; radius4 dilation; all uncovered cells mandatory. No labels or current costly features used.',
        confidence_protection='If enabled, union with previous OWN prediction max-anchor sigmoid >= guard threshold, aligned with the spatial state and expanded radius4. Uses pre-NMS confidence, no teacher/GT/current prediction; logs additional area. This may help via extra area alone, not yet a matched-area attribution.',
        recurrence='Four own post-activation maps plus matching retained descriptor warp together by metric SE2 nearest. Write only selected cells; unselected aligned complete activations persist. Save mixed state at current pose; full refresh period3/age/reset. Independent FM and deep anchor states for each mode; no teacher injections.',
        accounting='Actual pointwise output fraction (including dilation and full refresh), dense-basis area, executed conv+functional-linear MACsx2, same-GPU rotating model mean/P95, retained tensor bytes and peak CUDA allocated memory. Nonconv FLOPs not counted.',
        comparison='Compare selected to its own-approximate-recurrence masked_dense control, rank9 dense cache, and mature lowrank cache. Spatial reuse only is changed; old deep cache remains period3 for every mode. No training.'))
    write_json(out/'provenance.json', source)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'opencood/test', '-p', 'test_int*.py'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout); assert tests.returncode == 0
    arguments = json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train = relocate_tree(json.loads(Path(artifact_path(arguments['manifest'])).read_text())['train'])
    states = {k:None for k in modes}; control_state = None; costs = {}; mixing = basis_cost = None
    maximum = control_error = fusion_error = warp_error = storage_error = long_error = compact_error = 0.; writes = []
    _, stream = loader(train[:32], cfg, 0, 303)
    with torch.no_grad():
        for sample in stream:
            ego = to_device(sample['ego'], torch.device('cuda')); predictions = {}; details = {}
            for name, mode in modes.items():
                counter = ConvCounter(roots[name], torch.nn.ModuleList([split, repair]))
                previous = states[name]
                try: pred, states[name], info = mode.step(ego, states[name], sample['meta'])
                finally: counter.close()
                flag = bool(info['local_partial']); predictions[name] = pred; details[name] = info
                if flag: assert states[name]['cache']['value'] is previous['cache']['value']
                assert all(torch.isfinite(v).all() for v in pred.values())
                if name in ('base', 'dense'):
                    key = (name, flag)
                    if key in costs: assert costs[key] == counter.total
                    costs[key] = counter.total
                else:
                    cells = info['spatial_cells']
                    mixing = sum(cells*c.mix.weight.numel() for b in stage.blocks for c in (b.conv1, b.conv2))
                    basis_cost = sum(cells*(c.project if a.projected_run else c.basis).weight.numel() for b in stage.blocks for c in (b.conv1, b.conv2))
                    skipped = basis_cost if name in ('fused', 'prepared', 'fast_warp', 'cache_fp16', 'long_history', 'long_masked') else 0
                    if name in ('selected', 'periodic_selected', 'parent_selected', 'uniform_core', 'fused', 'prepared', 'fast_warp', 'cache_fp16', 'long_history') and not info['spatial_refresh']: skipped += mixing
                    if a.compact_state and (name in ('selected','periodic_selected') or (a.dependency_masks and name in ('parent_selected','uniform_core'))) and not info['spatial_refresh']:
                        skipped += basis_cost//2
                    if a.row_state and (name=='selected' or (a.row_plan and name=='parent_selected')) and not info['spatial_refresh']:skipped+=basis_cost//2
                    if name == 'selected' and not a.projected_run and not info['spatial_refresh'] and info['spatial_selected_cells'] == 0:
                        skipped += basis_cost
                    assert counter.total == costs[('dense', flag)]-skipped, (name, counter.total, skipped)
            for key in predictions['selected']:
                left, right = predictions['selected'][key], predictions['masked_dense'][key]
                maximum = max(maximum, float((left-right).abs().max()))
                assert torch.allclose(left, right, atol=1e-3, rtol=1e-3), maximum
            assert details['selected']['spatial_selected_cells'] == details['masked_dense']['spatial_selected_cells']
            if a.compact_state and not a.cell_age and not a.dependency_masks:
                for key in predictions['selected']:
                    compact_error = max(compact_error,float((predictions['selected'][key]-predictions['parent_selected'][key]).abs().max()))
                    assert torch.allclose(predictions['selected'][key],predictions['parent_selected'][key],atol=1e-3,rtol=1e-3)
                saved_bytes = 2*(128-stage.rank)*details['selected']['spatial_cells']*4
                assert details['parent_selected']['spatial_state_bytes']-details['selected']['spatial_state_bytes']==saved_bytes
            if a.single_state and not a.cell_age:
                for key in predictions['selected']:
                    compact_error=max(compact_error,float((predictions['selected'][key]-predictions['parent_selected'][key]).abs().max()))
                    assert torch.allclose(predictions['selected'][key],predictions['parent_selected'][key],atol=1e-3,rtol=1e-3)
                assert len(states['selected']['spatial']['maps'])==1
                assert details['parent_selected']['spatial_state_bytes']-details['selected']['spatial_state_bytes']==(0 if a.row_state else 32256000)
            if a.cell_age:
                for name in ('selected','masked_dense'):
                    assert details[name]['spatial_max_cell_age_s'] < .25
                    assert torch.isfinite(states[name]['spatial']['ages']).all()
                    assert details[name]['spatial_state_bytes']-details['periodic_selected']['spatial_state_bytes']==100800
            if a.fused:
                for key in predictions['fused']:
                    fusion_error = max(fusion_error, float((predictions['fused'][key]-predictions['selected'][key]).abs().max()))
                    assert torch.allclose(predictions['fused'][key], predictions['selected'][key], atol=1e-3, rtol=1e-3)
                assert details['fused']['spatial_selected_cells'] == details['selected']['spatial_selected_cells']
            if a.prepared:
                for key in predictions['prepared']:
                    assert torch.allclose(predictions['prepared'][key], predictions['fused'][key], atol=1e-3, rtol=1e-3)
                assert details['prepared']['spatial_selected_cells'] == details['fused']['spatial_selected_cells']
            if a.fast_warp:
                for key in predictions['fast_warp']:
                    warp_error = max(warp_error, float((predictions['fast_warp'][key]-predictions['prepared'][key]).abs().max()))
                    assert torch.equal(predictions['fast_warp'][key], predictions['prepared'][key]), 'Geometry execution changed prediction'
                for left,right in zip(states['prepared']['spatial']['maps'],states['fast_warp']['spatial']['maps']):
                    assert torch.equal(left,right), 'Geometry execution changed cached state'
                assert details['fast_warp']['spatial_selected_cells'] == details['prepared']['spatial_selected_cells']
            if a.cache_fp16:
                for key in predictions['cache_fp16']:
                    assert predictions['cache_fp16'][key].dtype == torch.float32
                    storage_error = max(storage_error, float((predictions['cache_fp16'][key]-predictions['prepared'][key]).abs().max()))
                for value in states['cache_fp16']['spatial']['maps']:
                    assert value.dtype == torch.float16 and torch.isfinite(value).all(), 'Half cache overflow'
                for key in ('reference','confidence'):
                    assert states['cache_fp16']['spatial'][key].dtype == torch.float32
                assert details['prepared']['spatial_state_bytes']-details['cache_fp16']['spatial_state_bytes'] == 25804800
            if a.long_history:
                for key in predictions['long_history']:
                    left,right = predictions['long_history'][key],predictions['long_masked'][key]
                    long_error = max(long_error,float((left-right).abs().max()))
                    assert torch.allclose(left,right,atol=2e-3,rtol=2e-3),long_error
                assert details['long_history']['spatial_selected_cells'] == details['long_masked']['spatial_selected_cells']
                assert details['long_history']['spatial_full_age'] < .45
                assert details['long_history']['spatial_state_bytes'] == details['cache_fp16']['spatial_state_bytes']
            writes.append(details['selected']['spatial_write_fraction'])
            pred, control_state, _ = control.step(ego, control_state, sample['meta'])
            for key in pred:
                control_error = max(control_error, float((pred[key]-predictions['dense'][key]).abs().max()))
                assert torch.allclose(pred[key], predictions['dense'][key], atol=1e-5, rtol=1e-5)
    exact_core_error=None
    if a.dependency_masks:
        with torch.no_grad():
            current=stage.observe(torch.randn(1,64,200,504,device='cuda'))
            target,maps=stage.dense_from_stem(current)
            old=[torch.randn_like(v)*3 for v in maps]
            core=torch.zeros_like(current[:,:1],dtype=torch.bool);core[:,:,20:30,40:50]=True;core[:,:,:2,:2]=True
            result,_,extra=stage.selected_from_stem(current,old,core)
            exact_core_error=float((result-target).masked_select(core).abs().max())
            assert torch.allclose(result.masked_select(core),target.masked_select(core),atol=1e-3,rtol=1e-3)
            assert torch.equal(result.masked_select(~core),old[-1].masked_select(~core))
            ks=[c for b in stage.blocks for c in (b.conv1,b.conv2)];ns=stage.last_layer_cells
            projection_macs=(sum(n*k.project.weight.numel() for n,k in zip(stage.last_projection_cells,ks)) if a.row_state
                else sum(ns[i]*ks[i+1].project.weight.numel() for i in (0,2)))
            assert extra==sum(n*k.mix.weight.numel() for n,k in zip(ns,ks))+projection_macs
    write_json(out/'engineering.json', dict(status='PASSED', frames=32, selected_vs_masked_dense_max_error=maximum,
        dependency_exact_current_core_error=exact_core_error,
        planned_vs_original_rows_max_error=compact_error if a.row_plan else None,
        row_vs_canvas_max_error=compact_error if a.row_state and not a.row_plan else None,
        single_vs_four_map_max_error=compact_error if a.single_state and not a.cell_age and not a.row_state else None,
        single_saved_state_bytes=32256000 if a.single_state and not a.cell_age and not a.row_state else None,
        compact_vs_original_state_max_error=compact_error if a.compact_state and not a.cell_age and not a.dependency_masks else None,
        cell_age_bound_and_bytes_verified=a.cell_age,
        fused_vs_factorized_max_error=fusion_error if a.fused else None,
        factorized_stage_parameters=sum(p.numel() for p in stage.parameters()),
        fused_stage_parameters=sum(p.numel() for p in roots['fused'].backbone.resnet.layer0.parameters()) if a.fused else None,
        prepared_packed_weight_bytes=sum(b.numel()*b.element_size() for n,b in roots['prepared'].named_buffers() if n.endswith('packed_weight')) if a.prepared else 0,
        fast_warp_exact_replay_max_error=warp_error if a.fast_warp else None,
        cache_fp16_vs_fp32_prediction_max_error=storage_error if a.cache_fp16 else None,
        long_history_vs_masked_dense_max_error=long_error if a.long_history else None,
        period1_vs_dense_max_error=control_error, mean_write_fraction=float(np.mean(writes)),
        constant_hook_macs={str(k):v for k,v in costs.items()}, dense_mix_macs=mixing, dense_basis_macs=basis_cost,
        costs_verified_against_actual_conv_hooks=True, own_immutable_deep_anchors=True))
    if a.readout_threshold is not None:
        from opencood.tools.evaluate_int_local_readout import evaluate_readout
        evaluate_readout(a,cfg,train,val,modes['selected'],out,costs,mixing+basis_cost)
        return
    if a.profile_frames:
        from opencood.tools.profile_int_basis_regions import profile_stream
        profile_stream(a, cfg, val, modes['selected'] if a.row_state else modes['fused'], modes['base'], out,
            modes['parent_selected'] if a.row_state else None)
        return
    if a.fast_warp:
        modes = {k:modes[k] for k in ('base', 'prepared', 'fast_warp')}
    if a.cache_fp16 and not a.long_history:
        modes = {k:modes[k] for k in ('base', 'prepared', 'cache_fp16')}
    if a.long_history:
        modes = {k:modes[k] for k in ('base', 'cache_fp16', 'long_history')}
    if a.audit_selection or a.verify_recurrence:
        from opencood.tools.int_selection_audit import capture_step, divergence_audit
    if a.audit_selection:
        modes = {k:modes[k] for k in ('selected','masked_dense')}
    states = {k:None for k in modes}; control_state = None
    stats = {k:{t:dict(tp=[], fp=[], gt=0, score=[]) for t in (.5, .7)} for k in modes}
    metrics = {k:dict(time=[], write=[], mix=[], layers=[], basis=[], projection=[], state=[], peak=[], full_age=[], cell_age=[], age_added=[], uncovered=[], priority=[], guard_added=[],
                     macs=0, refreshes=0, partials=0) for k in modes}
    dataset, stream = loader(val, cfg, 4, 303); fusion_mask_disagreements = prepared_mask_disagreements = warp_mask_disagreements = 0
    storage_mask_disagreements = 0
    recurrence_audits = []
    parent_mask_count_disagreements = 0
    with torch.no_grad():
        for index, sample in enumerate(stream):
            ego = to_device(sample['ego'], torch.device('cuda')); names = list(modes)
            offset = index % len(names); names = names[offset:]+names[:offset]; counts = {}
            previous_states = dict(states) if a.audit_selection or a.verify_recurrence else None
            audit_masks = {}
            for name in names:
                torch.cuda.synchronize(); before = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                if (a.audit_selection or a.verify_recurrence) and name in ('selected','masked_dense'):
                    pred, states[name], info, audit_masks[name] = capture_step(modes[name],ego,states[name],sample['meta'])
                else:
                    pred, states[name], info = modes[name].step(ego, states[name], sample['meta'])
                torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-tick)
                item = metrics[name]
                if index >= 10:
                    item['time'].append(elapsed)
                    item['peak'].append(max(0, torch.cuda.max_memory_allocated()-before))
                assert all(torch.isfinite(v).all() for v in pred.values())
                if name in ('cache_fp16','long_history'):
                    assert all(torch.isfinite(v).all() for v in states[name]['spatial']['maps']), 'Half cache overflow'
                    assert info['spatial_full_age'] < (.45 if name == 'long_history' else .25)
                flag = bool(info['local_partial']); item['partials'] += flag
                macs = costs[('base' if name == 'base' else 'dense', flag)]
                if name in ('fused', 'prepared', 'fast_warp', 'cache_fp16', 'long_history'): macs -= basis_cost
                if name in ('selected', 'periodic_selected', 'parent_selected', 'uniform_core', 'fused', 'prepared', 'fast_warp', 'cache_fp16', 'long_history') and not info['spatial_refresh']:
                    macs -= mixing
                    if a.compact_state and (name in ('selected','periodic_selected') or (a.dependency_masks and name in ('parent_selected','uniform_core'))): macs -= basis_cost//2
                    if a.row_state and (name=='selected' or (a.row_plan and name=='parent_selected')):macs-=basis_cost//2
                    if name == 'selected' and not a.projected_run and info['spatial_selected_cells'] == 0: macs -= basis_cost
                    macs += info['spatial_extra_linear_macs']
                item['macs'] += macs
                item['write'].append(info.get('spatial_write_fraction', 1.))
                item['mix'].append(info.get('spatial_mixer_executed_fraction', 1.))
                item['layers'].append(info.get('spatial_layer_update_fractions',info.get('spatial_layer_write_fractions',[info.get('spatial_write_fraction',1.)]*4)))
                item['basis'].append(info.get('spatial_basis_executed_fraction', 1.))
                item['projection'].append(info.get('spatial_projection_executed_fraction',1.))
                item['state'].append(info['state_bytes']+info['anchor_bytes']+info.get('spatial_state_bytes', 0))
                item['refreshes'] += int(info.get('spatial_refresh', True))
                item['full_age'].append(info.get('spatial_full_age', 0.))
                item['cell_age'].append(info.get('spatial_max_cell_age_s',0.))
                item['age_added'].append(info.get('spatial_age_added_fraction',0.))
                if a.cell_age and name in ('selected','masked_dense'):
                    assert info['spatial_max_cell_age_s'] < .25
                    assert torch.isfinite(states[name]['spatial']['ages']).all()
                item['uncovered'].append(info.get('spatial_uncovered_fraction', 0.))
                item['priority'].append(info.get('spatial_priority_fraction', 0.))
                item['guard_added'].append(info.get('spatial_guard_added_fraction', 0.))
                if name in ('selected', 'parent_selected', 'masked_dense', 'fused', 'prepared', 'fast_warp', 'cache_fp16'): counts[name] = info['spatial_selected_cells']
                if sample['meta']['supervised']:
                    data = dict(ego=ego); boxes, scores = dataset.post.post_process(data, dict(ego=pred)); gt = dataset.post.generate_gt_bbx(data)
                    for threshold in stats[name]: eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[name], threshold)
            if 'selected' in counts:
                if (a.audit_selection or a.verify_recurrence) and not torch.equal(audit_masks['selected'],audit_masks['masked_dense']):
                    audit = divergence_audit(modes['selected'],modes['masked_dense'],ego,previous_states,sample['meta'],audit_masks)
                    audit.update(scans=index+1,segment=sample['meta']['segment'],timestamp_us=int(sample['meta']['timestamp_us']))
                    if a.audit_selection:
                        write_json(out/'selection_audit.json',dict(audit,
                            scope='Diagnostic only. Independent own-state trajectories, first actual mask difference, then discarded same-input replay. No full AP or latency result.'))
                    else:
                        recurrence_audits.append(audit)
                        write_json(out/'recurrence_audits.json',dict(audits=recurrence_audits,
                            scope='Independent own-state masks may differ at floating-point thresholds. Every divergent frame gets discarded SAME-input backend validation; model/thresholds/trajectories unchanged.'))
                    assert audit['same_input_masks_equal'] and audit['same_input_numerically_close'], 'Same-input backend fidelity failed'
                    if a.audit_selection:
                        write_json(out/'status.json',dict(status='COMPLETED_DIAGNOSTIC',scans=index+1,found_divergence=True))
                        return
                    assert len(recurrence_audits) <= 128, 'Frequent trajectory divergence requires review'
                if not a.verify_recurrence:
                    assert counts['selected'] == counts['masked_dense'], 'Selection changed with execution backend'
            if 'fused' in counts: fusion_mask_disagreements += int(counts['fused'] != counts['selected'])
            if 'parent_selected' in counts: parent_mask_count_disagreements += int(counts['parent_selected'] != counts['selected'])
            if 'prepared' in counts and 'fused' in counts: prepared_mask_disagreements += int(counts['prepared'] != counts['fused'])
            if a.fast_warp: warp_mask_disagreements += int(counts['prepared'] != counts['fast_warp'])
            if a.cache_fp16 and not a.long_history: storage_mask_disagreements += int(counts['prepared'] != counts['cache_fp16'])
            if (index+1) % 200 == 0: write_json(out/'status.json', dict(status='RUNNING', phase='validation', scans=index+1))
    assert index+1 == len(val)
    if a.audit_selection:
        write_json(out/'selection_audit.json',dict(scans=len(val),found_divergence=False,scope='Diagnostic only; no full AP or timing result'))
        write_json(out/'status.json',dict(status='COMPLETED_DIAGNOSTIC',scans=len(val),found_divergence=False))
        return
    results = {}
    for name, item in metrics.items():
        results[name] = dict(ap70=eval_utils.calculate_ap(stats[name], .7)[0], ap50=eval_utils.calculate_ap(stats[name], .5)[0],
            mean_ms=float(np.mean(item['time'])), p95_ms=float(np.percentile(item['time'], 95)),
            conv_linear_gflops=2*item['macs']/len(val)/1e9,
            mean_write_fraction=float(np.mean(item['write'])), mean_executed_mix_area=float(np.mean(item['mix'])),
            mean_executed_basis_area=float(np.mean(item['basis'])), mean_state_bytes=float(np.mean(item['state'])),
            max_incremental_cuda_peak_bytes=max(item['peak']), mean_uncovered_fraction=float(np.mean(item['uncovered'])),
            mean_priority_fraction=float(np.mean(item['priority'])), mean_guard_added_fraction=float(np.mean(item['guard_added'])),
            max_spatial_full_age_s=max(item['full_age']), spatial_refresh_frames=item['refreshes'], deep_partial_frames=item['partials'],
            max_spatial_cell_age_s=max(item['cell_age']) if a.cell_age and name in ('selected','masked_dense') else None,
            mean_age_forced_additional_area=float(np.mean(item['age_added'])) if a.cell_age and name in ('selected','masked_dense') else None)
        results[name]['extra_packed_weight_bytes'] = sum(b.numel()*b.element_size() for n,b in roots[name].named_buffers() if n.endswith('packed_weight'))
        if a.dependency_masks:
            results[name]['mean_layer_update_fractions']=np.mean(item['layers'],axis=0).tolist()
            results[name]['mean_layer_state_write_fractions']=([float(np.mean(item['write']))] if a.single_state and (name in ('selected','masked_dense','periodic_selected') or (a.row_state and name=='parent_selected'))
                else np.mean(item['layers'],axis=0).tolist())
        results[name]['static_geometry_bytes'] = modes[name].warp.geometry_bytes if name == 'fast_warp' else 0
        if a.projected_run:
            results[name]['mean_executed_projection_area'] = float(np.mean(item['projection'])) if name != 'base' else None
            # This architecture has input projections, not depthwise spatial bases.
            results[name]['mean_executed_basis_area'] = None
    assert abs(results['base']['ap70']-.7555532616222492) < 1e-8
    if 'dense' in results: assert abs(results['dense']['ap70']-sb['cached_ap70']) < 1e-8
    if a.dependency_masks and not a.cell_age:
        expected=({.5:.7553159775482435,.75:.7541971676152598} if a.single_state else {.5:.7557311115176569,.75:.7546151539838869})[a.threshold]
        assert abs(results['parent_selected']['ap70']-expected)<1e-7
    if a.single_state and a.cell_age:
        expected={.5:.7553159775482435,.75:.7541971676152598}[a.threshold]
        assert abs(results['periodic_selected']['ap70']-expected)<1e-7
    write_json(out/'report.json', dict(status='COMPLETED', scans=len(val), labels=sum(r['supervised'] for r in val), results=results,
        recurrence_verified=a.verify_recurrence,
        single_spatial_state=a.single_state,
        row_spatial_execution=a.row_state,
        planned_row_execution=a.row_plan,
        row_plan_note='Overrides parent wording: parent_selected is original active-row execution. Selected shares one per-frame geometry plan and pre-packs fixed kernels; same masks/weights/arithmetic/MACs/state. Extra packed weight bytes are reported separately; index arrays and temporary activations remain timed and included in peak.' if a.row_plan else None,
        row_note='Overrides other parent descriptions: parent_selected is SAME single-H policy with full temporary canvases and two dense projections. Selected uses active rows with four local projections4/3/2/1 plus3x3 supports3/2/1/0. Same persistent H, weights, refresh and core. Gather/index work is timed; only conv/linear MACs counted as FLOPs. No new training.' if a.row_state else None,
        single_state_note='Overrides generic parent/state wording: selected and masked_dense retain only final128 H; prior intermediate maps replaced by zero temporary canvases. Without cell_age, parent_selected is FOUR-map dependency model with SAME weights/core policy/refresh. With cell_age, periodic_selected is SAME single-H period3 control, age expiry enters final core before upstream3/2/1/0 supports. No uniform_core mode. Four layer update fractions and one persistent map write fraction separate; temporary allocations inside model time.' if a.single_state else None,
        dependency_guard=a.dependency_guard,
        dependency_note='Overrides generic shared-mask/parent-state scope: final core is undilated content/uncovered union plus own-confidence support(radius4 if dependency_guard, else0); four supports dilate3/2/1/0. Recomputed final core has complete current-stem dependencies; cached outside final core remains approximate. Parent_selected is compact original shared radius4 baseline; uniform_core shares final-core selection policy but no upstream support halo. Same checkpoint and refresh; per-layer state-write and actual weighted execution areas separate.' if a.dependency_masks else None,
        cell_age_note=('Single-H selected/masked_dense use .25s direct-age expiry, no global periodic refresh; expired cells receive full CURRENT-stem dependencies via3/2/1/0 supports. Periodic_selected is same single-H period3 control. No old intermediate feature dependencies, but nearest-motion warp/stale unselected regions and downstream spatial mixing remain approximate. Deep cache unchanged.' if a.single_state else 'Selected/masked_dense use local .25s direct-age expiry, no global periodic refresh. Periodic_selected retains compact period3 baseline; deep cache unchanged. Direct cache-age bound does not bound accumulated feature-dependency error.') if a.cell_age else None,
        compact_state_parent_mask_count_disagreement_frames=parent_mask_count_disagreements if a.compact_state and not a.cell_age else None,
        compact_state_note='Selected uses projected intermediate(r) and full block output(128) pairs; parent_selected caches four128 maps. Same compressed weights. Dense full function unchanged. Projection area averages two dense and two selected projections. Compare AP/state bytes/actual MACs independently; nearest warp and zero-bias projection commute in exact arithmetic, floating paths may differ.' if a.compact_state else None,
        own_mask_disagreement_frames=len(recurrence_audits) if a.verify_recurrence else None,
        own_mask_xor_cells_total=sum(v['own_mask_xor_cells'] for v in recurrence_audits) if a.verify_recurrence else None,
        same_input_replay_passed_all=all(v['same_input_masks_equal'] and v['same_input_numerically_close'] for v in recurrence_audits) if a.verify_recurrence else None,
        recurrence_timing_note='Mask capture overhead included in selected/control timing; comparisons and discarded shared-input replays excluded. No pure-backend latency equivalence claimed.' if a.verify_recurrence else None,
        fused_mask_count_disagreement_frames=fusion_mask_disagreements if 'fused' in modes else None,
        prepared_mask_count_disagreement_frames=prepared_mask_disagreements if 'fused' in modes and 'prepared' in modes else None,
        fast_warp_mask_count_disagreement_frames=warp_mask_disagreements if a.fast_warp else None,
        cache_fp16_mask_count_disagreement_frames=storage_mask_disagreements if a.cache_fp16 and not a.long_history else None,
        scope=('Projected-input spatial kernels; dense cheap 1x1 projection plus selected expensive 3x3 output rows, four own aligned FP32 activation caches. Same compressed checkpoint dense and masked-dense controls; mature uncompressed base separately. Period3/.25s and own previous confidence guard; no training/teacher injection. ' if a.projected_run else '')+'Actual own-recursive four-map spatial cache, whole aligned activation reuse, selected output-column mixing. Fixed heuristic threshold, no training. Dense bases (unless fused or replaced by projected-input kernels), stem/current FM and existing readout/deep cache included. Fused execution composes full-rank kernels and eliminates explicit spatial bases, with same cache/mask semantics. Areas include refresh and dilated requests, no duplicated halo outputs. Conv plus functional-linear MACsx2 excludes nonconv arithmetic. Same GPU rotating wall time includes warp/selection/gather/scatter; excludes IO/H2D/postprocess. State tensor bytes are per-mode; incremental CUDA peaks measure transient allocation above live multi-mode allocations, not standalone process VRAM.'))
    write_json(out/'status.json', dict(status='COMPLETED'))


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'readout-run', 'repair-run', 'memory-run', 'output'): p.add_argument('--'+key, required=True)
    spatial = p.add_mutually_exclusive_group(required=True)
    spatial.add_argument('--basis-run')
    spatial.add_argument('--projected-run', help='Completed input-channel projection recovery run, fixed best-by-fullAP')
    p.add_argument('--threshold', type=float, choices=(.25, .5, .75), required=True)
    p.add_argument('--confidence-guard', type=float, choices=(.1, .3), default=None)
    p.add_argument('--fused', action='store_true', help='Include equivalent composed-kernel selected execution')
    p.add_argument('--prepared', action='store_true', help='Also compare shared neighborhood indices and prepacked weights')
    p.add_argument('--fast-warp', action='store_true', help='Validate base/prepared/cached-geometry warp after full engineering controls')
    p.add_argument('--cache-fp16', action='store_true', help='Validate FP16 history storage with FP32 computation')
    p.add_argument('--long-history', action='store_true', help='Compare spatial period5/.45s to period3/.25s with identical FP16 storage')
    p.add_argument('--profile-frames', type=int, default=0, help='Bounded timing diagnosis instead of AP validation (64..512 frames)')
    p.add_argument('--audit-selection', action='store_true', help='Projected only: stop at first mask divergence, save same-input replay; no full AP result')
    p.add_argument('--verify-recurrence', action='store_true', help='Projected only: full AP with logged mask divergences and same-input verification on each')
    p.add_argument('--compact-state', action='store_true', help='Cache projected intermediates, paired with original four-map state')
    p.add_argument('--cell-age', action='store_true', help='Compact state: compare .25s per-cell expiry against periodic full refresh')
    p.add_argument('--dependency-masks',action='store_true',help='Compact state: nested3/2/1/0 supports, paired original radius4 and uniform radius0')
    p.add_argument('--dependency-guard',action='store_true',help='Dependency core: restore original confidence radius4, keep content undilated')
    p.add_argument('--single-state',action='store_true',help='Guarded dependency update: retain only final H, paired four-map parent')
    p.add_argument('--row-state',action='store_true',help='Single-H temporary active rows, paired same single-H canvas parent')
    p.add_argument('--row-plan',action='store_true',help='Prepared row geometry and fixed kernels, paired original row execution')
    p.add_argument('--readout-threshold',type=float,choices=(.25,.5),default=None,
                   help='Fixed four-layer threshold.5 baseline plus local complete readout feature reuse')
    p.add_argument('--readout-period',type=int,choices=(3,5),default=3,
                   help='Readout only; period5/.45s is paired with unchanged period3/.25s')
    p.add_argument('--readout-audit-frames',type=int,default=0,
                   help='Training-prefix phase/selector diagnosis only, no AP or model latency')
    p.add_argument('--readout-backend-frames',type=int,default=0,
                   help='Same-state training-prefix row/tile8/tile16/masked benchmark, no AP')
    p.add_argument('--readout-selector-frames',type=int,default=0,
                   help='Training-prefix energy/fragmentation and discarded one-step mask audit')
    p.add_argument('--readout-ranking-audit',action='store_true',
                   help='Within selector diagnosis compare score rankings under equal executed-MAC budget')
    p.add_argument('--readout-budget-scale',type=float,choices=(.8,1.),default=None,
                   help='Absolute input-error selection under own-relative mask executed cost budget')
    p.add_argument('--gpu', type=int, required=True); p.add_argument('--worker', action='store_true')
    p.add_argument('--readout-ranking',choices=('absolute','relative'),default='absolute',
                   help='Input-error ranking under the same actual row-MAC budget')
    p.add_argument('--readout-calibrate-frames',type=int,choices=(0,512),default=0,
                   help='Train-only fixed threshold fit to aggregate80 percent partial readout MACs')
    p.add_argument('--readout-calibration',default=None,
                   help='Completed train calibration.json; freeze its scalar before full validation')
    p.add_argument('--stem-audit-frames',type=int,choices=(0,512),default=0,
                   help='Train-only pre-stem descriptor and six-layer dependency-cost feasibility')
    p.add_argument('--deep-audit-frames',type=int,choices=(0,512),default=0,
                   help='Train-only dependency-cost feasibility of periodic deep local refresh')
    p.add_argument('--recurrent-deep-epochs',type=int,default=0,
                   help='Train complete recursive D updater; maximum1..24 epochs, zero disables')
    p.add_argument('--recurrent-deep-period',type=int,choices=(3,6),default=6)
    p.add_argument('--recurrent-deep-resume',default=None,help='Completed-epoch latest.pth to resume in new output directory')
    p.add_argument('--path-map',default=None,help='Explicit JSON source-prefix to host-prefix mapping; preserve artifact bytes/contracts')
    p.add_argument('--recurrent-three-checkpoint',default=None,help='Frozen period3-trained repair for paired recurrence comparison')
    p.add_argument('--recurrent-six-checkpoint',default=None,help='Frozen period6-trained repair; evaluate at periods3/4/6')
    p.add_argument('--recurrent-local-reference',default=None,help='Completed frozen-period comparison used to fix checkpoint and dense AP')
    p.add_argument('--recurrent-local-period',type=int,choices=(3,4,6),default=4)
    p.add_argument('--recurrent-readout-calibration',default=None,help='Compose frozen train-calibrated readout reuse with complete D recurrence')
    p.add_argument('--recurrent-readout-profile-frames',type=int,choices=(0,512),default=0,help='Bounded same-state row batching and stage timing on train prefix')
    p.add_argument('--recurrent-readout-layout',action='store_true',help='Full independent comparison of row batching and packed aligned state')
    p.add_argument('--short-readout-epochs',type=int,default=0,help='Train complete readout short recurrence, maximum budget')
    p.add_argument('--short-readout-width',type=int,choices=(32,64),default=32)
    p.add_argument('--short-readout-kind',choices=('residual','current','current_gated'),default='residual')
    p.add_argument('--short-readout-resume',default=None)
    p.add_argument('--y-only-epochs',type=int,default=0,help='Y-only spatial state; no H/D on ordinary frames')
    p.add_argument('--y-only-width',type=int,choices=(64,96),default=64)
    p.add_argument('--y-only-period',type=int,choices=(3,5),default=3)
    p.add_argument('--y-only-resume',default=None,help='Exact Y-only optimizer/schedule continuation')
    p.add_argument('--short-readout-train-period',type=int,choices=(3,5),default=3)
    p.add_argument('--short-readout-init',default=None,help='Weights-only warm start; fresh optimizer/schedule, distinct from exact resume')
    p.add_argument('--short-readout-diagnostic',default=None,help='Frozen repair checkpoint for offline512 train-frame history diagnosis')
    p.add_argument('--current-readout-local',default=None,help='Frozen current-only repair checkpoint for selected local execution')
    p.add_argument('--current-readout-local-calibration',default=None,help='Existing train-only readout calibration to transfer unchanged')
    p.add_argument('--current-readout-profile',action='store_true',help='Same-state512 train frames: row batches and separate intrusive phase timing')
    p.add_argument('--current-readout-geometry',action='store_true',help='Add independent fixed-grid geometry reuse comparison to full local evaluation')
    p.add_argument('--current-readout-packed',action='store_true',help='Add independent packed complete-state comparison to full local evaluation')
    p.add_argument('--current-readout-period5',action='store_true',help='Frozen current network: five-frame dense/local readout controls, no change to upstream H/D')
    a = p.parse_args(); out = Path(a.output)
    if not 0<=a.y_only_epochs<=36:p.error('Y-only budget must be0..36')
    if a.y_only_resume and not a.y_only_epochs:p.error('Y-only resume requires training')
    if a.y_only_epochs and (not a.recurrent_local_reference or a.recurrent_local_period!=4 or
            a.short_readout_epochs or a.short_readout_resume or a.short_readout_diagnostic or a.short_readout_init or
            a.current_readout_local or a.recurrent_readout_calibration):
        p.error('Y-only training requires frozen reference without another experiment')
    if bool(a.current_readout_local)!=bool(a.current_readout_local_calibration):p.error('Local current readout requires checkpoint and calibration')
    if a.current_readout_profile and not a.current_readout_local:p.error('Current readout profile requires frozen local checkpoint')
    if a.current_readout_geometry and (not a.current_readout_local or a.current_readout_profile):p.error('Geometry comparison requires full frozen current readout evaluation')
    if a.current_readout_packed and (not a.current_readout_local or a.current_readout_profile):p.error('Packed comparison requires full frozen current readout evaluation')
    if a.current_readout_period5 and (not a.current_readout_local or a.current_readout_profile):p.error('Period5 comparison requires full frozen current readout evaluation')
    if a.current_readout_local and (a.short_readout_epochs or a.short_readout_resume or a.short_readout_diagnostic or not a.recurrent_local_reference or a.recurrent_local_period!=4 or a.recurrent_readout_calibration):
        p.error('Local current readout requires frozen four-frame reference without other readout experiment')
    if a.short_readout_diagnostic and (a.short_readout_epochs or a.short_readout_resume or not a.recurrent_local_reference or a.recurrent_local_period!=4 or a.recurrent_readout_calibration):
        p.error('Short readout diagnostic requires frozen four-frame D reference without training/composition')
    if not 0<=a.short_readout_epochs<=48:p.error('Short readout budget must be0..48')
    if a.short_readout_resume and not a.short_readout_epochs:p.error('Resume requires short readout training')
    if (a.short_readout_init or a.short_readout_train_period!=3) and (not a.short_readout_epochs or a.short_readout_kind!='current'):
        p.error('Period adaptation requires current-only short readout training')
    if a.short_readout_init and a.short_readout_resume:p.error('Weights-only initialization cannot be combined with exact resume')
    if a.short_readout_epochs and (not a.recurrent_local_reference or a.recurrent_readout_calibration or a.recurrent_local_period!=4):
        p.error('Short readout training requires frozen four-frame D reference without readout composition')
    if a.recurrent_readout_calibration and not a.recurrent_local_reference:
        p.error('Readout composition requires frozen recurrent reference')
    if a.recurrent_readout_profile_frames and not a.recurrent_readout_calibration:
        p.error('Readout backend profile requires frozen composition')
    if a.recurrent_readout_layout and (not a.recurrent_readout_calibration or a.recurrent_readout_profile_frames):
        p.error('Packed layout comparison requires frozen composition without prefix profiling')
    if a.recurrent_local_reference and (a.recurrent_three_checkpoint or a.recurrent_six_checkpoint or a.recurrent_deep_epochs or a.recurrent_deep_resume or
            a.deep_audit_frames or a.stem_audit_frames or a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_calibrate_frames or a.readout_calibration or a.readout_budget_scale is not None or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames or a.readout_ranking_audit):
        p.error('Local recurrent comparison requires frozen period reference and fixed H baseline only')
    if bool(a.recurrent_three_checkpoint)!=bool(a.recurrent_six_checkpoint):p.error('Both frozen recurrence checkpoints required')
    if a.recurrent_three_checkpoint and (a.recurrent_deep_epochs or a.recurrent_deep_resume or a.deep_audit_frames or a.stem_audit_frames or
            a.readout_period!=3 or a.readout_threshold!=.5 or a.readout_calibrate_frames or a.readout_calibration or a.readout_budget_scale is not None or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames or a.readout_ranking_audit):
        p.error('Frozen recurrence comparison requires fixed H baseline without training or other diagnostics')
    if a.recurrent_deep_resume and not a.recurrent_deep_epochs:p.error('Resume requires recurrent D training')
    if not 0<=a.recurrent_deep_epochs<=24:p.error('Recurrent D budget must be0..24')
    if a.recurrent_deep_epochs and (a.deep_audit_frames or a.stem_audit_frames or a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_calibrate_frames or a.readout_calibration or a.readout_budget_scale is not None or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames or a.readout_ranking_audit):
        p.error('Recurrent D training requires fixed H baseline without other diagnostics')
    if a.deep_audit_frames and (a.stem_audit_frames or a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_calibrate_frames or a.readout_calibration or a.readout_budget_scale is not None or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames or a.readout_ranking_audit):
        p.error('Deep audit requires original period3/threshold.5 without other diagnostics')
    if a.stem_audit_frames and (a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_calibrate_frames or a.readout_calibration or a.readout_budget_scale is not None or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames or a.readout_ranking_audit):
        p.error('Stem audit requires original period3/threshold.5 without other diagnostics')
    if a.readout_calibrate_frames or a.readout_calibration:
        if (a.readout_calibrate_frames and a.readout_calibration or a.readout_period!=3 or
                a.readout_threshold!=.5 or a.readout_budget_scale is not None or a.readout_ranking_audit or
                a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames):
            p.error('Fixed calibration requires period3/threshold.5 and no other selector diagnostics/budget')
    if a.readout_ranking!='absolute' and a.readout_budget_scale is None:
        p.error('Relative ranking control requires explicit readout budget')
    if a.readout_budget_scale is not None and (a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_selector_frames or a.readout_backend_frames or a.readout_audit_frames):
        p.error('Budgeted readout requires period3/threshold.5 full validation without diagnostic modes')
    if a.readout_ranking_audit and not a.readout_selector_frames:
        p.error('Ranking diagnosis requires readout-selector-frames')
    if a.readout_selector_frames and (a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_audit_frames or a.readout_backend_frames or not 64<=a.readout_selector_frames<=512):
        p.error('Selector audit requires period3/threshold.5 and 64..512 scans without other diagnostics')
    if a.readout_backend_frames and (a.readout_period!=3 or a.readout_threshold!=.5 or
            a.readout_audit_frames or not 64<=a.readout_backend_frames<=512):
        p.error('Readout backend profile requires period3/threshold.5 and 64..512 scans, without phase audit')
    if a.readout_audit_frames and (a.readout_period!=5 or not 64<=a.readout_audit_frames<=512):
        p.error('Readout phase diagnosis requires period5 and 64..512 training scans')
    if a.readout_period==5 and a.readout_threshold!=.5:
        p.error('Long readout history requires the verified threshold.5 readout candidate')
    if a.readout_threshold is not None and (not a.row_plan or a.threshold!=.5 or
            a.confidence_guard!=.1 or a.profile_frames or a.cell_age):
        p.error('Local readout requires fixed planned-row threshold.5/guard.1 period3 baseline')
    if a.row_plan and not a.row_state:
        p.error('Planned rows require row-state')
    if a.row_state and (not a.single_state or a.cell_age):
        p.error('Row execution requires single-state periodic comparison without cell age')
    if a.single_state and not a.dependency_guard:
        p.error('Single-state comparison requires --dependency-guard')
    if a.dependency_guard and not a.dependency_masks:
        p.error('Independent confidence support requires --dependency-masks')
    if a.dependency_masks and (not a.compact_state or (a.cell_age and not a.single_state) or a.threshold not in (.5,.75)):
        p.error('Dependency supports require compact verified state, threshold.5/.75; cell age only for single-state')
    if a.cell_age and not a.compact_state:
        p.error('Cell-age comparison requires compact projected state')
    if a.compact_state and (not a.projected_run or not a.verify_recurrence or a.audit_selection):
        p.error('Compact projected state requires projected run and verified full recurrence')
    if a.audit_selection and not a.projected_run:
        p.error('Selection diagnosis requires --projected-run')
    if a.verify_recurrence and (not a.projected_run or a.audit_selection):
        p.error('Verified recurrence requires projected full validation, without diagnostic early stop')
    if a.projected_run and (a.fused or a.prepared or a.fast_warp or a.cache_fp16 or a.long_history or (a.profile_frames and not a.row_state) or a.confidence_guard is None):
        p.error('Projected comparison requires confidence guard and plain FP32 four-mode validation')
    if a.profile_frames and (not (a.fused or a.row_state) or not 64 <= a.profile_frames <= 512):
        p.error('Profiling requires --fused or --row-state and 64..512 frames')
    if a.prepared and (not a.fused or a.profile_frames):
        p.error('Prepared comparison requires --fused and full validation')
    if a.fast_warp and not a.prepared:
        p.error('Fast warp requires --prepared')
    if a.cache_fp16 and (not a.prepared or a.fast_warp or a.confidence_guard is None):
        p.error('FP16 storage requires --prepared and confidence guard, without --fast-warp')
    if a.long_history and not a.cache_fp16:
        p.error('Long spatial history requires --cache-fp16')
    if a.worker:
        limit=28800 if a.recurrent_deep_epochs or a.short_readout_epochs or a.y_only_epochs else 14400
        def expired(s, f): raise TimeoutError('Basis experiment exceeded its wall-clock budget')
        signal.signal(signal.SIGALRM, expired); signal.alarm(limit)
        try: worker(a)
        except Exception as exc: write_json(out/'status.json', dict(status='FAILED', error=repr(exc))); raise
        finally: signal.alarm(0)
        return
    free, util = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']).decode().strip().split(',')
    if a.short_readout_epochs or a.short_readout_diagnostic or a.current_readout_local or a.y_only_epochs:
        # Observed process footprint about5.5GiB; leave launch headroom. Other
        # processes and nonzero utilization are permitted by the user.
        if int(free)<8192:raise RuntimeError('Short readout needs at least8192MiB free at launch')
    elif int(free) < 22000 or int(util) > 5: raise RuntimeError('GPU occupied')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = [sys.executable, '-m', 'opencood.tools.evaluate_int_basis_regions']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out/'launch.json', dict(pid=proc.pid, gpu=a.gpu, command=command, created_unix=time.time(), timeout_seconds=28800 if a.recurrent_deep_epochs or a.short_readout_epochs or a.y_only_epochs else 14400,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid, output=str(out))))


if __name__ == '__main__': main()
