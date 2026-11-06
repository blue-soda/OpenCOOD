#!/usr/bin/env python
"""Official CoSDH hybrid evaluation with strict loading and auditable exports.

Run in an isolated compatible environment. No source edits and no training.
limit=0 means full split; default limit=2 is only a smoke test, never a main AP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def checkpoint_path(folder):
    best = sorted(folder.glob('net_epoch_bestval_at*.pth'))
    if best:
        if len(best) != 1:
            raise ValueError('Ambiguous best checkpoints: {}'.format(best))
        return best[0]
    numbered = []
    for path in folder.glob('net_epoch*.pth'):
        match = re.fullmatch(r'net_epoch(\d+)\.pth', path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError('No official net_epoch*.pth checkpoint in {}'.format(folder))
    return max(numbered)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', required=True, help='JSON path, absolute or relative to data-root')
    parser.add_argument('--output', required=True, help='A new output directory; existing paths rejected')
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--gpu', default='0', help='Single physical CUDA device index or UUID')
    parser.add_argument('--wire-codec',action='store_true',help='Audited demand/latent/late-box actual transport; diagnostic compute overhead included')
    parser.add_argument('--comm-range', type=float, default=None, help='Explicit participant range override in metres; omitted preserves official config')
    parser.add_argument('--compute-smoke-reference', help='Immutable full-wire reference; four-frame parity gate only')
    parser.add_argument('--compute-count',type=int,choices=(4,64),default=4)
    parser.add_argument('--compute-repeat',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--ego-only', action='store_true', help='Same collaborative weights, only ego participant; no remote features or late boxes')
    parser.add_argument('--feature-only', action='store_true', help='Keep both intermediate feature participants, omit roadside late detections')
    parser.add_argument('--count-thresholds', help='Training-only fitted count thresholds; diagnostic tensor transport only')
    parser.add_argument('--count-steps', type=int, choices=(8, 32))
    parser.add_argument('--count-event-quotas', type=int, nargs=3, help='Three fixed per-stage IF event caps')
    parser.add_argument('--count-wire', action='store_true', help='Actual integer counts, scales, masks and request; no late boxes')
    parser.add_argument('--readout-thresholds', help='Training-only post-fusion calibration')
    parser.add_argument('--readout-mode', choices=('identity', 'clip', 'count4', 'count16'))
    parser.add_argument('--membrane-mode', choices=('identity', 'fixed', 'local'))
    parser.add_argument('--front-thresholds')
    parser.add_argument('--front-mode', choices=('identity', 'clip', 'count16', 'count32'))
    parser.add_argument('--front-initial-voltage', type=float, choices=(0., .5), default=0.)
    parser.add_argument('--second-thresholds')
    parser.add_argument('--second-mode', choices=('identity','clip','count32'))
    parser.add_argument('--second-initial-voltage', type=float, choices=(0.,.5), default=0.)
    parser.add_argument('--third-thresholds')
    parser.add_argument('--third-mode', choices=('identity','clip','count32'))
    parser.add_argument('--third-initial-voltage', type=float, choices=(0.,.5), default=0.)
    parser.add_argument('--pillar-thresholds')
    parser.add_argument('--pillar-mode', choices=('identity','clip','count32'))
    parser.add_argument('--pillar-initial-voltage', type=float, choices=(0.,.5), default=0.)
    parser.add_argument('--selection-mode', choices=('identity','request_only','confidence_only','geometry_only'))
    parser.add_argument('--geometry-backend',action='store_true',help='Single encoder pass; remove confidence and request computation')
    parser.add_argument('--synapse-mode',choices=('none','stage','channel'),default='none')
    parser.add_argument('--spiking-io',action='store_true',help='Load the converted 1x1 spiking sender/receiver checkpoint')
    parser.add_argument('--synapse-identity',action='store_true',help='Fixed-weight ablation: replace learned current gains by one')
    parser.add_argument('--suppress-remote-current',action='store_true',help='Frozen geometry-model ablation: retain local membrane and readout, set aligned remote current to zero; not independently trained ego or wire accounting')
    args = parser.parse_args()
    if args.synapse_mode != 'none' and (not args.geometry_backend or args.membrane_mode != 'local'):
        parser.error('Learned synapses require the geometry/local membrane route')
    if args.synapse_identity and args.synapse_mode == 'none':
        parser.error('Identity ablation requires a registered learned synapse')
    if args.suppress_remote_current and (not args.geometry_backend or args.membrane_mode!='local' or args.count_wire or args.wire_codec):
        parser.error('Remote-current ablation requires the local-membrane geometry backend without wire accounting')
    if args.geometry_backend and args.selection_mode!='geometry_only':
        parser.error('Streamlined backend requires the fixed geometry-only selection study')
    if args.selection_mode and (args.pillar_mode!='count32' or args.pillar_initial_voltage!=.5 or
                               ((args.count_wire or args.count_event_quotas is not None) and not args.geometry_backend)):
        parser.error('Selection study fixes half-init PFN Count32; count wire/quotas require the geometry backend')
    if bool(args.pillar_thresholds)!=bool(args.pillar_mode):
        parser.error('Pillar thresholds and mode required together')
    if args.pillar_mode and (args.third_mode!='count32' or args.third_initial_voltage!=0.):
        parser.error('Pillar study retains zero-initialized Count32 third stage')
    if args.pillar_initial_voltage and args.pillar_mode!='count32':
        parser.error('Half-threshold pillar requires count32')
    if bool(args.third_thresholds)!=bool(args.third_mode):
        parser.error('Third-stage thresholds and mode required together')
    if args.third_mode and (args.second_mode!='count32' or args.second_initial_voltage!=.5):
        parser.error('Third-stage study preserves the half-initialized Count32 second stage')
    if args.third_initial_voltage and args.third_mode!='count32':
        parser.error('Half-threshold third stage requires count32')
    if bool(args.second_thresholds)!=bool(args.second_mode):
        parser.error('Second-stage thresholds and mode required together')
    if args.second_mode and (args.front_mode!='count32' or args.front_initial_voltage!=0.):
        parser.error('Second-stage study preserves the zero-initialized Count32 first stage')
    if args.second_initial_voltage and args.second_mode!='count32':
        parser.error('Half-threshold second stage requires count32')
    if args.front_initial_voltage and args.front_mode not in ('count16','count32'):
        parser.error('Half-threshold initialization requires a count frontend')
    if bool(args.front_thresholds) != bool(args.front_mode):
        parser.error('Front stage thresholds and mode required together')
    if args.front_mode and not args.spiking_io and (args.membrane_mode != 'local' or args.readout_mode != 'count16'):
        parser.error('Front stage probe requires the fixed local membrane / Count16 route')
    if args.membrane_mode and not args.spiking_io and args.readout_mode not in ('count4', 'count16'):
        parser.error('Membrane probe requires calibrated count readout')
    if bool(args.readout_thresholds) != bool(args.readout_mode):
        parser.error('Readout mode and thresholds must be specified together')
    if args.readout_mode and not args.count_thresholds:
        parser.error('Readout probe requires the fixed count sender')
    if bool(args.count_thresholds) != bool(args.count_steps):
        parser.error('Count thresholds and steps must be specified together')
    if args.count_thresholds and (not args.feature_only or args.wire_codec or args.compute_smoke_reference):
        parser.error('Count diagnostic requires feature-only without the FP16 wire codec or compute benchmark')
    if (args.count_wire or args.count_event_quotas is not None) and not args.count_thresholds:
        parser.error('Count wire/quotas require count representation')
    if args.feature_only and (args.ego_only or args.compute_smoke_reference):
        parser.error('feature-only is distinct from ego-only and the hybrid compute reference')
    if args.ego_only:
        if args.wire_codec or args.compute_smoke_reference:
            parser.error('ego-only has zero transport and cannot use wire/compute adapters')
        if args.comm_range not in (None, 1e-6):
            parser.error('ego-only uses the audited 1e-6 metre participant range')
        args.comm_range = 1e-6
    if args.compute_smoke_reference and (args.limit != 0 or not args.wire_codec):
        parser.error('Compute gate requires --limit 0 --wire-codec')
    compute_reference = Path(args.compute_smoke_reference).resolve() if args.compute_smoke_reference else None
    if args.limit < 0 or ',' in args.gpu:
        parser.error('limit must be >=0 and gpu must name one device')
    repo = Path(args.repo).resolve()
    model_dir = Path(args.model_dir).resolve()
    data_root = Path(args.data_root).resolve()
    split = Path(args.split)
    split = split.resolve() if split.is_absolute() else (data_root / split).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    start_wall = time.time()
    write_json(output / 'status.json', {'stage': 'initializing', 'started_unix': start_wall, 'args': vars(args)})
    try:
        import numpy as np
        import torch
        import opencood
        from torch.utils.data import DataLoader, Subset
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils, inference_utils
        from opencood.data_utils.datasets import build_dataset
        from opencood.utils import eval_utils

        package_file = Path(opencood.__file__).resolve()
        if repo not in package_file.parents:
            raise RuntimeError('Wrong opencood package: {}'.format(package_file))
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this benchmark; CPU fallback is forbidden')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.25, device=0)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(303)
        np.random.seed(303)
        torch.manual_seed(303)
        torch.cuda.manual_seed_all(303)
        device = torch.device('cuda:0')

        config_path = model_dir / 'config.yaml'
        cfg = yaml_utils.load_yaml(str(config_path))
        # Only dataset locations and the explicitly requested official inference mode.
        overrides = {'data_dir': str(data_root), 'root_dir': str(data_root / 'train.json'),
                     'validate_dir': str(split), 'test_dir': str(split)}
        if args.comm_range is not None:
            if not np.isfinite(args.comm_range) or args.comm_range <= 0:
                raise ValueError('comm-range must be finite and positive')
            overrides['comm_range'] = args.comm_range
        cfg.update(overrides)
        original_fusion = cfg['fusion']['core_method']
        cfg['fusion']['core_method'] = 'intermediatelate'
        if 'box_align' in cfg:
            raise RuntimeError('Unexpected box_align external files: resolve explicitly before reproduction')
        checkpoint = checkpoint_path(model_dir)
        model = train_utils.create_model(cfg)
        from cosdh_synapse import install as install_synapse
        install_synapse(model,args.synapse_mode,args.readout_thresholds)
        if args.spiking_io:
            from cosdh_spiking_io import install as install_spiking
            install_spiking(model,args.readout_thresholds)
        saved = torch.load(str(checkpoint), map_location='cpu')
        expected = model.state_dict()
        if not isinstance(saved, dict):
            raise TypeError('Expected official raw state_dict checkpoint')
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        shape_mismatches = {k: {'expected': list(expected[k].shape),
                                'actual': list(saved[k].shape) if hasattr(saved[k], 'shape') else 'not tensor'}
                            for k in set(expected) & set(saved)
                            if not isinstance(saved[k], torch.Tensor) or expected[k].shape != saved[k].shape}
        load_audit = {'missing': missing, 'unexpected': unexpected, 'shape_mismatches': shape_mismatches}
        write_json(output / 'load_audit.json', load_audit)
        if missing or unexpected or shape_mismatches:
            raise RuntimeError('Strict checkpoint audit failed; see load_audit.json')
        if any(not torch.isfinite(v).all().item() for v in saved.values() if torch.is_floating_point(v)):
            raise ValueError('Nonfinite checkpoint tensor')
        model.load_state_dict(saved, strict=True)
        del saved
        model.to(device).eval()
        if args.synapse_identity: model.snn_synapse.force_identity = True
        parameters = sum(p.numel() for p in model.parameters())
        counts = None
        if args.count_thresholds:
            from cosdh_counts import CoSDHCounts
            counts = CoSDHCounts(model, args.count_thresholds, args.count_steps, args.count_event_quotas)
        readout = None
        if args.readout_mode:
            from cosdh_readout import CoSDHReadout
            readout = CoSDHReadout(model, args.readout_thresholds, args.readout_mode)
        membrane = None
        if args.membrane_mode:
            from cosdh_membrane import CoSDHMembrane
            membrane_steps = int(args.readout_mode[5:]) if args.readout_mode in ('count4', 'count16') else 16
            membrane = CoSDHMembrane(model, args.readout_thresholds, args.membrane_mode, membrane_steps, suppress_remote=args.suppress_remote_current)
        front = None
        if args.front_mode:
            from cosdh_front_stage import CoSDHFrontStage
            front = CoSDHFrontStage(model, args.front_thresholds, args.front_mode, initial_voltage=args.front_initial_voltage,encoder_paths=1 if args.geometry_backend else 2)
        second=None
        if args.second_mode:
            second=CoSDHFrontStage(model,args.second_thresholds,args.second_mode,initial_voltage=args.second_initial_voltage,encoder_paths=1 if args.geometry_backend else 2)
            assert second.stage==1 and front.stage==0
        third=None
        if args.third_mode:
            third=CoSDHFrontStage(model,args.third_thresholds,args.third_mode,initial_voltage=args.third_initial_voltage,encoder_paths=1 if args.geometry_backend else 2)
            assert third.stage==2
        pillar=None
        if args.pillar_mode:
            from cosdh_pillar import CoSDHPillar
            pillar=CoSDHPillar(model,args.pillar_thresholds,args.pillar_mode,args.pillar_initial_voltage,encoder_paths=1 if args.geometry_backend else 2)
        selection=None
        if args.selection_mode and not args.geometry_backend:
            from cosdh_selection import CoSDHSelection
            selection=CoSDHSelection(model,args.selection_mode)
        geometry=None
        if args.geometry_backend:
            from cosdh_geometry import GeometryBackend
            geometry=GeometryBackend(model)
        if not args.wire_codec and not args.count_wire:
            import zlib
            def forbidden(*args, **kwargs): raise AssertionError('Ordinary evaluation must not use codec')
            zlib.compress = zlib.decompress = forbidden

        dataset = build_dataset(cfg, visualize=True, train=False)
        wire=None
        if args.wire_codec:
            from cosdh_wire import CoSDHWire
            wire=CoSDHWire(model,dataset,output,include_late_boxes=not args.feature_only)
        if args.count_wire:
            if args.geometry_backend:
                from cosdh_geometry_wire import CoSDHGeometryCountWire
                wire=CoSDHGeometryCountWire(model,counts,output)
            else:
                from cosdh_count_wire import CoSDHCountWire
                wire=CoSDHCountWire(model,counts,output)
        ids = list(dataset.split_info)
        source_ids = json.loads(split.read_text(encoding='utf-8'))
        if ids != source_ids or len(ids) != len(dataset):
            raise RuntimeError('Dataset did not preserve requested ordered split')
        count = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
        if count == 0:
            raise RuntimeError('Empty evaluation split')
        loader = DataLoader(Subset(dataset, list(range(count))), batch_size=1, num_workers=0,
                            collate_fn=dataset.collate_batch_test, shuffle=False,
                            pin_memory=False, drop_last=False)
        try:
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        except Exception:
            commit = None
        metadata = {'repo': str(repo), 'repo_commit': commit, 'opencood_file': str(package_file),
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                    'source_config': str(config_path), 'source_config_sha256': digest(config_path),
                    'split': str(split), 'split_sha256': digest(split), 'ordered_split_ids': ids,
                    'data_root': str(data_root), 'overrides': overrides,
                    'original_fusion': original_fusion, 'fusion_method': 'intermediatelate',
                    'limit': args.limit, 'evaluated_samples_planned': count, 'dataset_samples': len(dataset),
                    'metric': 'official BEV polygon IoU AP with global prediction score sort',
                    'gt_parity_with_our_snn': 'NOT_ESTABLISHED', 'actual_communication_bytes': None,
                    'seed': 303, 'parameters': parameters, 'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                    'physical_gpu_selection': args.gpu, 'allocator_fraction': .25,
                    'warmup_forwards_first_batch': 3,
                    'timing_scope': 'CUDA-synchronized official hybrid inference incl both model forwards and postprocess/NMS, excludes loading/H2D/export/AP; batch1',
                    'source_script_sha256': digest(__file__), 'started_unix': start_wall}
        write_json(output / 'provenance.json', metadata)
        metadata['ego_only'] = args.ego_only
        metadata['feature_only'] = args.feature_only
        metadata['readout'] = dict(mode=args.readout_mode, thresholds_sha256=digest(args.readout_thresholds),
            interface='post_fusion', original_fusion_preserved=True, membrane_conversion=False) if readout else None
        metadata['membrane'] = dict(mode=args.membrane_mode, single_frame=True,
            added_parameters=sum(p.numel() for p in model.snn_synapse.parameters()) if args.synapse_mode!='none' else 0,
            synapse=model.snn_synapse.statistics() if args.synapse_mode!='none' else None,
            gate='none' if args.membrane_mode == 'identity' else ('theta/(theta+local)' if args.membrane_mode == 'local' else '0.5'),
            max_replaced=args.membrane_mode != 'identity') if membrane else None
        if membrane and args.membrane_mode != 'identity':
            metadata['readout'].update(original_fusion_preserved=False, membrane_conversion=True)
        metadata['front_stage'] = dict(mode=args.front_mode,initial_voltage_threshold_fraction=args.front_initial_voltage,thresholds_sha256=digest(args.front_thresholds),
            scope='First plain BEV stage four ReLUs; both agents and confidence/communication paths; remaining encoder ANN') if front else None
        metadata['second_stage']=dict(mode=args.second_mode,initial_voltage_threshold_fraction=args.second_initial_voltage,
            thresholds_sha256=digest(args.second_thresholds),scope='Six second-stage ReLUs, both agents and both paths; PillarVFE/final BEV stage ANN') if second else None
        if second:
            metadata['front_stage']['scope']='First-stage four IF ReLUs retained; second-stage intervention reported separately; PillarVFE/final stage ANN'
        metadata['third_stage']=dict(mode=args.third_mode,initial_voltage_threshold_fraction=args.third_initial_voltage,
            thresholds_sha256=digest(args.third_thresholds),scope='Nine third-stage ReLUs, both agents and both paths; PillarVFE and ANN detection backend retained') if third else None
        if third:
            metadata['front_stage']['scope']='Four first-stage IFs retained; later stages reported separately; PillarVFE remains ANN'
            metadata['second_stage']['scope']='Six second-stage IFs retained; third stage reported separately; PillarVFE remains ANN'
        metadata['pillar']=dict(mode=args.pillar_mode,initial_voltage_threshold_fraction=args.pillar_initial_voltage,
            thresholds_sha256=digest(args.pillar_thresholds),scope='PFN point activation before max, both agents, shared input to both BEV paths; ANN decoding and detection backend retained') if pillar else None
        if pillar:
            for key in ('front_stage','second_stage','third_stage'):
                metadata[key]['scope']='BEV IF stage retained; PFN intervention reported separately; ANN decoding and detection backend retained'
        metadata['selection']=dict(mode=args.selection_mode,scope='Receiver confidence/request mask ablation, counts unchanged; no byte advantage claim') if selection else None
        metadata['geometry_backend']=dict(enabled=True,encoder_paths=1,confidence_and_request_computation_removed=True) if geometry else None
        metadata['count_conversion'] = dict(steps=args.count_steps, thresholds_sha256=digest(args.count_thresholds),
            stage_quotas=args.count_event_quotas,wire=args.count_wire,
            scope='Frozen remote-only decoder-input IF counts; original masks/decoder/fusion; optional crossing-time quotas and integer wire') if counts else None
        metadata['single_agent_boundary'] = 'Collaborative checkpoint with remote participants disabled; not independently trained single-agent' if args.ego_only else None
        metadata['wire_codec']='demand_sparse_latent_dependency_closure_late_boxes_v1' if args.wire_codec else None
        if args.count_wire:metadata['wire_codec']='sparse_integer_count_scale_mask_request_v1'
        if args.count_wire and geometry:
            metadata['wire_codec']='geometry_oneway_sparse_integer_count_scale_v1'
            metadata['count_conversion']['scope']='Remote decoder-input counts; receiver-derived geometric support; no request, confidence mask or late boxes'
        if args.wire_codec and args.feature_only:
            metadata['wire_codec']='demand_sparse_latent_dependency_closure_v1'
        metadata['wire_compute_scope']='includes redundant diagnostic decode/warp, not optimized algorithmic cost' if args.wire_codec else None
        write_json(output / 'provenance.json', metadata)
        if compute_reference:
            from compute_cosdh_adapter import smoke
            try:smoke(model,dataset,wire,ids,compute_reference,output,metadata,args.gpu,loader,args.compute_count,args.compute_repeat)
            finally:wire.close()
            return
        stat = {t: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for t in (.3, .5, .7)}
        rows = []
        # Store output references, and inspect only AFTER timing to avoid finite-check synchronization overhead.
        raw_outputs = []
        hook = model.register_forward_hook(lambda module, inputs, result: raw_outputs.append(result))

        def finite_tree(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all().item())
            if isinstance(value, dict):
                return all(finite_tree(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_tree(v) for v in value)
            return True

        def array(tensor, empty_shape):
            return np.empty(empty_shape, dtype=np.float32) if tensor is None else tensor.detach().cpu().numpy()

        iterator = iter(loader)
        with torch.no_grad():
            for index in range(count):
                t0 = time.perf_counter()
                batch = next(iterator)
                load_seconds = time.perf_counter() - t0
                if batch is None:
                    raise RuntimeError('Dataset returned None at {}'.format(index))
                if int(batch['ego']['sample_idx']) != index:
                    raise RuntimeError('Frame index mismatch')
                t0 = time.perf_counter()
                if args.ego_only:
                    # Dataset range filtering preserves ego processing and cooperative GT.
                    # Drop the hybrid roadside dictionary as well as its intermediate input.
                    batch = {'ego': batch['ego']}
                    if batch['ego']['record_len'].tolist() != [1]:
                        raise ValueError('ego-only requires exactly one participant')
                if args.feature_only:
                    # The ego dictionary already contains both agents' features.
                    # Removing only the outer road entry disables its separate
                    # detector forward and late boxes, preserving intermediate fusion.
                    batch = {'ego': batch['ego']}
                    if batch['ego']['record_len'].tolist() != [2]:
                        raise ValueError('feature-only requires both intermediate participants')
                batch = train_utils.to_device(batch, device)
                if wire:wire.begin(batch,str(ids[index]))
                torch.cuda.synchronize()
                h2d_seconds = time.perf_counter() - t0
                if index == 0:
                    for _ in range(3):
                        raw_outputs.clear()
                        if wire:wire.begin(batch,str(ids[index]))
                        if pillar:pillar.begin()
                        if selection:selection.begin()
                        if geometry:geometry.begin()
                        warmup = inference_utils.inference_late_fusion(batch, model, dataset)
                        if pillar:pillar.finish()
                        if selection:selection.finish()
                        if geometry:geometry.finish()
                        if not finite_tree(warmup) or not finite_tree(raw_outputs):
                            raise ValueError('Nonfinite warmup output')
                    del warmup
                    raw_outputs.clear()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                raw_outputs.clear()
                if wire:wire.begin(batch,str(ids[index]))
                if counts:counts.begin()
                if readout:readout.begin()
                if membrane:membrane.begin()
                if front:front.begin()
                if second:second.begin()
                if third:third.begin()
                if pillar:pillar.begin()
                if selection:selection.begin()
                if geometry:geometry.begin()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = inference_utils.inference_late_fusion(batch, model, dataset)
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - t0
                if not finite_tree(result) or not finite_tree(raw_outputs):
                    raise ValueError('Nonfinite model/output at {}'.format(ids[index]))
                pred = array(result['pred_box_tensor'], (0, 8, 3))
                score = array(result['pred_score'], (0,))
                gt = array(result['gt_box_tensor'], (0, 8, 3))
                if pred.shape != (len(score), 8, 3) or gt.shape[1:] != (8, 3):
                    raise ValueError('Unexpected box/score shape')
                for threshold in stat:
                    eval_utils.caluclate_tp_fp(result['pred_box_tensor'], result['pred_score'],
                                              result['gt_box_tensor'], stat, threshold)
                frame_path = output / 'frames' / ('{:06d}.npz'.format(index))
                np.savez_compressed(str(frame_path), frame_id=np.asarray(str(ids[index])),
                                    pred=pred, score=score, gt=gt)
                row = {'index': index, 'frame_id': str(ids[index]), 'prediction_count': len(score),
                       'gt_count': len(gt), 'file': str(frame_path.relative_to(output)),
                       'sha256': digest(frame_path), 'inference_seconds': inference_seconds,
                       'data_loading_seconds': load_seconds, 'h2d_seconds': h2d_seconds,
                       'model_forwards': len(raw_outputs),
                       'record_len': batch['ego']['record_len'].detach().cpu().tolist(),
                       'gt_array_sha256': hashlib.sha256(gt.tobytes(order='C')).hexdigest()}
                if args.ego_only:
                    if row['model_forwards'] != 1:
                        raise ValueError('ego-only hybrid inference executed a remote forward')
                    row['actual_communication_bytes'] = 0
                if args.feature_only and row['model_forwards'] != 1:
                    raise ValueError('feature-only executed a separate roadside detector')
                if counts:
                    row['count_conversion'] = counts.finish()
                    row['count_conversion']['codec_executed'] = args.count_wire
                if readout:row['readout'] = readout.finish()
                if membrane:
                    row['membrane'] = membrane.finish()
                    row['readout'].update(original_fusion_preserved=args.membrane_mode=='identity', membrane_conversion=args.membrane_mode!='identity')
                if front:row['front_stage'] = front.finish()
                if second:row['second_stage']=second.finish()
                if third:row['third_stage']=third.finish()
                if pillar:row['pillar']=pillar.finish()
                if selection:row['selection']=selection.finish()
                if geometry:row['geometry_backend']=geometry.finish()
                rows.append(row)
                if wire:row['wire']=wire.finish(index)
                with (output / 'frames.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                if (index + 1) % 25 == 0 or index == count - 1:
                    write_json(output / 'status.json', {'stage': 'evaluating', 'completed': index + 1,
                                                       'planned': count, 'last_frame_id': str(ids[index]),
                                                       'updated_unix': time.time()})
                    print('completed {}/{}'.format(index + 1, count), flush=True)
        hook.remove()
        if wire:wire.close()
        if readout:readout.close()
        if membrane:membrane.close()
        if front:front.close()
        if second:second.close()
        if third:third.close()
        if pillar:pillar.close()
        if selection:selection.close()
        if geometry:geometry.close()
        if counts:
            counts.close()
            original = torch.load(str(checkpoint), map_location='cpu')
            assert all(torch.equal(v.cpu(), original[k]) for k, v in model.state_dict().items())
        ap = {str(t): float(eval_utils.calculate_ap(stat, t)[0]) for t in stat}
        timings = np.asarray([r['inference_seconds'] for r in rows])
        summary = {'status': 'completed', 'full_split': count == len(dataset),
                   'smoke_only': count < len(dataset), 'samples': count,
                   'gt_boxes': stat[.7]['gt'], 'pred_boxes': sum(r['prediction_count'] for r in rows),
                   'bev_ap': ap, 'parameters': parameters,
                   'inference_ms_mean': float(timings.mean() * 1000),
                   'inference_ms_median': float(np.median(timings) * 1000),
                   'inference_ms_p95': float(np.percentile(timings, 95) * 1000),
                   'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                   'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                   'all_outputs_finite': True, 'actual_communication_bytes': None,
                   'gt_parity_with_our_snn': 'NOT_ESTABLISHED',
                   'frames_manifest_sha256': digest(output / 'frames.jsonl'),
                   'completed_unix': time.time(), 'wall_seconds': time.time() - start_wall}
        write_json(output / 'result_stat.json', stat)
        if wire:
            def stats(values):
                return {'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'max':float(np.max(values)),'count':len(values)}
            summary['wire_statistics']={k:stats([r['wire'][k] for r in rows]) for k in ('demand_bytes','response_bytes','total_bytes','envelope_bytes','late_boxes')}
            summary['actual_communication_bytes']=summary['wire_statistics']['total_bytes']['mean']
            summary['wire_note']='Actual binary demand, three dependency-closed FP16 latent/mask sections and pre-NMS late boxes; 160-byte two-way metadata included; link/IP overhead excluded; diagnostic redundant compute not suitable for efficiency comparison'
        summary['ego_only'] = args.ego_only
        summary['feature_only'] = args.feature_only
        summary['count_conversion'] = metadata['count_conversion']
        summary['readout'] = metadata['readout']
        summary['membrane'] = metadata['membrane']
        summary['front_stage'] = metadata['front_stage']
        summary['second_stage']=metadata['second_stage']
        summary['third_stage']=metadata['third_stage']
        summary['pillar']=metadata['pillar']
        summary['selection']=metadata['selection']
        summary['geometry_backend']=metadata['geometry_backend']
        if counts:summary['model_state_unchanged'] = True
        if args.feature_only and args.wire_codec:
            summary['wire_note']='Actual demand and three dependency-closed FP16 latent/mask sections; no late boxes; 160-byte two-way metadata included; link/IP overhead excluded'
        if args.count_wire:
            summary['wire_note']='Actual sparse integer counts with per-channel FP32 scales, three exact masks, one reused request and 160B metadata; no FP16 features/late boxes; link/IP overhead excluded'
            if geometry:
                summary['wire_note']='Actual one-way three-stage sparse integer counts, indices, FP32 channel scales, 144B FP64 pose/frame envelope and 8B container; geometric support derived locally; no request, transmitted mask or late boxes; link/IP overhead excluded; diagnostic codec timing is not an efficiency claim'
        if args.ego_only:
            summary['actual_communication_bytes'] = 0
            summary['single_agent_frames'] = len(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', summary)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    except Exception:
        write_json(output / 'failure.json', {'failed_unix': time.time(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
