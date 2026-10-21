"""Paired IF-prefix versus complete-voxel selection at 2048 actual bytes.

Frozen model and identical source encoding per frame. First-overflow policy,
full metadata, no training. Complete rows are offline full-frame scheduling.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
from opencood.utils.snapshot_selection import select_snapshot
from opencood.tools.profile_e3dsnn_event_granularity import encode_snapshot

SOURCE_SHA = '60b7e0ff84e992862902aa9325b56ac13ab5658d70c4aacb194028638f5105dc'
MODES = ('events', 'rows')


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(root, reference):
    new=json.loads((root/'summary.json').read_text())
    old=json.loads((reference/'summary.json').read_text())
    assert new['passed'] and old['passed'] and len(new['records'])==len(old['records'])==128
    for a,b in zip(new['records'],old['records']):
        assert {k:v for k,v in a.items() if k!='selection_seconds'} == {k:v for k,v in b.items() if k!='selection_seconds'}
    write(root/'reference_equivalence.json',dict(passed=True,records=128,
        reference_path=str(reference/'summary.json'),reference_sha256=sha(reference/'summary.json'),
        optimized_sha256=sha(root/'summary.json'),all_non_timing_fields_identical=True))


def cache_audit(root, cache):
    root.mkdir(parents=True, exist_ok=False)
    status = root / 'status.json'
    try:
        source_report = json.loads((cache.parent / 'summary.json').read_text())
        assert source_report['passed'] and source_report['checkpoint_sha256'] == SOURCE_SHA
        rows = []
        for i in range(64):
            source = cache / ('{:04d}.npz'.format(i))
            with np.load(str(source)) as data:
                ids = data['addresses']; meta = json.loads(str(data['metadata']))
            for mode in MODES:
                packet, counts, report = select_snapshot(ids, meta, 2048, mode)
                assert packet == encode_snapshot(counts, meta)
                if mode == 'events':
                    expected = np.bincount(ids[:report['accepted_units']].astype(np.int64),
                        minlength=counts.size).reshape(counts.shape)
                    assert np.array_equal(expected, counts)
                report.update(frame=i, frame_id=meta['contract']['receiver_frame'],
                    source_sha256=sha(source), snapshot_sha256=hashlib.sha256(packet).hexdigest(),
                    dense_sparse_packet_exact=True, decode_exact=True)
                rows.append(report)
            if (i+1) % 8 == 0:
                write(status, dict(state='running', frames=i+1))
                print(json.dumps(dict(frames=i+1)), flush=True)
        means = {mode: {k: float(np.mean([r[k] for r in rows if r['mode']==mode]))
            for k in ('actual_bytes','unused_bytes','selected_events','selected_voxels','selection_seconds')}
            for mode in MODES}
        # Bound the next stage's reference cost before reserving GPU runtime.
        gate = sum(means[m]['selection_seconds'] for m in MODES) <= 3.5
        write(root / 'summary.json', dict(passed=True, records=rows, means=means,
            checkpoint_sha256=SOURCE_SHA, full_evaluation_cost_gate=gate,
            cost_gate='mean combined CPU selection <=3.5 seconds/frame; not deployment latency',
            semantics='same frozen cached source, first-overflow policy, no AP or model updates'))
        write(status, dict(state='complete', full_evaluation_cost_gate=gate))
        print(json.dumps(means), flush=True)
    except Exception:
        write(status, dict(state='failed', error=traceback.format_exc()))
        raise


def run(root, source, audit, limit=0):
    import torch
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.tools.train_e3dsnn_single import write_json, seed_worker
    from opencood.tools.launch_e3dsnn_hybrid_round import evaluation
    from opencood.utils import eval_utils, eval_3d_utils
    from opencood.utils.e3dsnn_transplant import encoder_digest
    from opencood.models.fuse_modules.e3dsnn_event_stream import integrate_and_fire, stream_readout
    from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract
    report = json.loads((audit / 'summary.json').read_text())
    assert report['passed'] and report['full_evaluation_cost_gate']
    assert report['checkpoint_sha256'] == SOURCE_SHA
    equivalence=json.loads((audit/'reference_equivalence.json').read_text())
    assert equivalence['passed'] and equivalence['all_non_timing_fields_identical']
    assert equivalence['optimized_sha256']==sha(audit/'summary.json')
    assert equivalence['reference_sha256']==sha(Path(equivalence['reference_path']))
    provenance = Path(__file__).resolve().parents[2] / 'docs/cp_snn/results/event_granularity_20261021/sha256.json'
    expected_cache = json.loads(provenance.read_text())['cache_files']
    assert len(report['records']) == 128
    for row in report['records']:
        assert row['source_sha256'] == expected_cache['{:04d}.npz'.format(row['frame'])]
        assert row['dense_sparse_packet_exact'] and row['decode_exact'] and row['actual_bytes'] <= 2048
    root.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint = source / 'train/best_ap70.pth'; config = source / 'b4096.yaml'
        assert sha(checkpoint) == SOURCE_SHA
        random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929)
        torch.cuda.manual_seed_all(20260929); torch.cuda.set_per_process_memory_fraction(.2)
        cfg = load_yaml(str(config)); dataset = build_dataset(cfg, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=dataset.collate_batch_test, worker_init_fn=seed_worker,
            multiprocessing_context='spawn')
        model = train_utils.create_model(cfg).cuda().eval()
        initial = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(initial, strict=True); model.requires_grad_(False)
        assert model.event_max_spikes==8 and model.event_packet_events==256
        encoder = encoder_digest(model); basis = model.event_basis_digest()
        stats = {m: {t: dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for m in MODES}
        stats3d = copy.deepcopy(stats); frames = {m: [] for m in MODES}; audit_rows = []
        commit = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
        protocol = dict(checkpoint_sha256=SOURCE_SHA, budget_bytes=2048, modes=MODES,
            selection='first serialized overflow stops; no monotonicity or maximality claim',
            shared_source=True, training=False, audit_sha256=sha(audit/'summary.json'),
            seed=20260929, full_frames=1738, full_gt=28622, limit=limit, code_commit=commit,
            limitation='complete rows use future within-frame counts; offline scheduling, not online causality',
            exit='one matched selection contrast; no parameter sweep if row selection has no benefit')
        write_json(root/'protocol.json',protocol)
        for mode in MODES:
            folder=root/mode; (folder/'predictions').mkdir(parents=True); (folder/'packets').mkdir()
            write_json(folder/'manifest_selection.json', dict(protocol, checkpoint=str(checkpoint),
                config_sha256=sha(config), data_manifest_sha256=sha(Path(cfg['data_manifest'])),
                split_sha256={k:sha(Path(cfg[k])) for k in ('root_dir','validate_dir')},
                selection_mode=mode, cuda_allocator_memory_fraction=.2))
        started=time.time()
        with torch.no_grad():
            for i,batch in enumerate(loader):
                if limit and i >= limit: break
                if time.time()-started>14400: raise TimeoutError('Four hour paired inference cap exceeded')
                batch=train_utils.to_device(batch,torch.device('cuda')); data=batch['ego']
                _,membrane=model.encode_membrane(data['processed_lidar'],1)
                road=model.encode_lidar(data['road_lidar'],1).dense()
                stream=integrate_and_fire(torch.relu(model.content.project(road)),8)
                contract=MessageContract(str(data['frame_ids'][0]),str(data['road_frame_ids'][0]),
                    'infrastructure',encoder,'event-content32-'+basis,
                    tuple(int(x) for x in membrane.shape[2:]),tuple(model.origin),tuple(model.step),
                    tuple(int(x) for x in data['timestamps_us'][0].cpu().tolist()))
                pose=data['road_to_ego'];meta=dict(contract=asdict(contract),source_to_ego=pose[0].cpu().tolist(),
                    channels=32,max_spikes=8,schema='if-unit-events-v1')
                source_digest=hashlib.sha256(stream.addresses.tobytes()).hexdigest()
                fid=str(dataset.data[i]); assert fid==contract.receiver_frame
                for mode in MODES:
                    packet,decoded,record=select_snapshot(stream.addresses,meta,2048,mode)
                    assert packet==encode_snapshot(decoded,meta)
                    if mode=='events': direct=stream.prefix(record['accepted_units'])
                    else:
                        full=stream.prefix(len(stream.addresses))
                        mask=torch.as_tensor(np.any(decoded!=0,axis=1,keepdims=True),device=full.device)
                        direct=full*mask
                    counts=membrane.new_tensor(decoded); assert torch.equal(counts,direct)
                    state=stream_readout(membrane,counts,pose,contract,model.joint_receiver,model.content.decode)
                    reference=stream_readout(membrane,direct,pose,contract,model.joint_receiver,model.content.decode)
                    assert torch.equal(state,reference)
                    output=model.detect_bev(state.flatten(1,2)); direct_output=model.detect_bev(reference.flatten(1,2))
                    assert all(torch.equal(output[k],direct_output[k]) for k in ('psm','rm','dm'))
                    boxes,scores,gt=dataset.post_process_no_fusion(batch,{'ego':output})
                    for threshold in stats[mode]:
                        eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[mode],threshold)
                    eval_3d_utils.calculate_tp_fp_3d(boxes,scores,gt,stats3d[mode])
                    pred=boxes.cpu().numpy() if boxes is not None else np.empty((0,8,3),dtype=np.float32)
                    score=scores.cpu().numpy() if scores is not None else np.empty(0,dtype=np.float32)
                    assert np.isfinite(pred).all() and np.isfinite(score).all()
                    folder=root/mode;path=folder/'predictions'/('{:06d}.npz'.format(i))
                    np.savez_compressed(str(path),frame_id=np.asarray(fid),pred_boxes=pred,scores=score,gt_boxes=gt.cpu().numpy())
                    packet_path=folder/'packets'/('{:06d}.bin'.format(i));packet_path.write_bytes(packet)
                    frames[mode].append(dict(index=i,frame_id=fid,file=path.name,sha256=sha(path),
                        predicted_boxes=len(pred),gt_boxes=len(gt),
                        communication=dict(road_to_vehicle_packet_bytes=[len(packet)],total_bytes=len(packet))))
                    record.update(index=i,frame_id=fid,source_events_sha256=source_digest,
                        packet_sha256=sha(packet_path),counts_state_predictions_exact=True)
                    audit_rows.append(record)
                if (i+1)%50==0:
                    write_json(root/'status.json',dict(state='running',frames=i+1,total=len(dataset)))
                    print(json.dumps(dict(frames=i+1)),flush=True)
        assert len(audit_rows)==2*(limit or 1738) and len(dataset)==1738
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        assert sha(checkpoint)==SOURCE_SHA
        write_json(root/'selection_audit.json',dict(passed=True,all_state_unchanged=True,records=audit_rows))
        results={}
        for mode in MODES:
            folder=root/mode; rows=frames[mode];lengths=[r['communication']['total_bytes'] for r in rows]
            assert max(lengths)<=2048
            write_json(folder/'predictions/index.json',dict(schema=1,frames=rows,order='validation loader order'))
            metrics=dict(samples=len(rows),indexed_samples=len(dataset),full_split=len(rows)==len(dataset),
                predicted_boxes=sum(r['predicted_boxes'] for r in rows),gt_boxes=sum(r['gt_boxes'] for r in rows),
                ap={str(t):eval_utils.calculate_ap(stats[mode],t)[0] for t in stats[mode]},
                ap_3d={str(t):eval_utils.calculate_ap(stats3d[mode],t)[0] for t in stats3d[mode]},
                metric='BEV and upright 3D AP; paired shared source snapshot selection',
                predictions=dict(directory='predictions',frames=len(rows),index_sha256=sha(folder/'predictions/index.json')),
                feature_packets=dict(count=len(rows),mean_bytes=float(np.mean(lengths)),median_bytes=float(np.median(lengths)),
                    min_bytes=min(lengths),max_bytes=max(lengths),p95_bytes=float(np.percentile(lengths,95)),
                    total_bytes=sum(lengths),scope='serialized payload and full metadata; excludes network framing and retransmission'))
            write_json(folder/'evaluation.json',metrics)
            write_json(root/'status.json',dict(state='cpu_replay',mode=mode))
            with (root/(mode+'_replay.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(folder)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
            if limit:
                replay=json.loads((folder/'offline_replay.json').read_text())
                assert replay['passed'] and replay['exact_match'] and replay['all_file_hashes_verified']
                results[mode]=dict(evaluation=metrics,replay=replay)
            else:
                results[mode]=evaluation(folder)
        write_json(root/'summary.json',dict(passed=True,protocol=protocol,results=results,
            audit_sha256=sha(root/'selection_audit.json'),all_state_unchanged=True))
        write_json(root/'status.json',dict(state='complete',summary_sha256=sha(root/'summary.json')))
    except Exception:
        write_json(root/'status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'))
        raise


def coordinate(root, source, audit, gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    try:
        for name,limit in (('smoke',2),('full',0)):
            write(root/(name+'_admission.json'),check_gpu(gpu,True,.2))
            write(root/'pipeline_status.json',dict(state='running',stage=name))
            with (root/(name+'.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_snapshot_selection',
                    '--root',str(root/name),'--source',str(source),'--audit',str(audit),'--limit',str(limit)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=21600)
            assert json.loads((root/name/'summary.json').read_text())['passed']
        write(root/'pipeline_status.json',dict(state='complete',summary_sha256=sha(root/'full/summary.json')))
    except Exception:
        write(root/'pipeline_status.json',dict(state='failed',error=traceback.format_exc(),note='No automatic retry'))
        raise


def launch(root, source, audit, gpu):
    from opencood.tools.launch_e3dsnn_joint_round import check_gpu
    from opencood.tools.launch_e3dsnn_hybrid_round import REPO
    check=json.loads((audit/'summary.json').read_text())
    assert check['passed'] and check['full_evaluation_cost_gate']
    assert json.loads((audit/'reference_equivalence.json').read_text())['passed']
    assert sha(source/'train/best_ap70.pth')==SOURCE_SHA
    if root.exists(): raise FileExistsError('Fresh launch directory required')
    admission=check_gpu(gpu,True,.2)
    root.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'pipeline.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','opencood.tools.evaluate_e3dsnn_snapshot_selection',
            '--root',str(root),'--source',str(source),'--audit',str(audit),'--gpu',str(gpu),'--coordinate'],
            cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,gpu=gpu,admission=admission,audit_sha256=sha(audit/'summary.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        started_unix=time.time(),protocol='2-frame real smoke plus CPU replay then paired 1738-frame frozen comparison')
    write(root/'launch.json',record); print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--cache');p.add_argument('--source');p.add_argument('--audit')
    p.add_argument('--verify-reference')
    p.add_argument('--launch',action='store_true');p.add_argument('--coordinate',action='store_true')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--limit',type=int,choices=(0,2),default=0)
    a=p.parse_args()
    if a.launch: launch(Path(a.root),Path(a.source),Path(a.audit),a.gpu)
    elif a.coordinate: coordinate(Path(a.root),Path(a.source),Path(a.audit),a.gpu)
    elif a.verify_reference: verify_reference(Path(a.root),Path(a.verify_reference))
    elif a.cache: cache_audit(Path(a.root),Path(a.cache))
    else: run(Path(a.root),Path(a.source),Path(a.audit),a.limit)
