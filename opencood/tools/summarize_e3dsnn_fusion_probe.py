"""Export verified same-device ablations and descriptive GT-region probes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np


def evidence(path, content=True):
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if content:
        result['content'] = json.loads(path.read_text())
    return result


def aggregate(rows):
    valid = [row for row in rows if row['supported_cells'] > 0]
    if not valid:
        return dict(total_regions=len(rows), observed_regions=0)
    cells = sum(row['supported_cells'] for row in valid)
    return dict(total_regions=len(rows), observed_regions=len(valid), supported_cells=cells,
        metrics={key: dict(mean_across_regions=float(np.mean([row[key] for row in valid])),
                          cell_weighted_mean=sum(row[key] * row['supported_cells'] for row in valid) / cells)
                 for key in valid[0] if key != 'supported_cells'})


def summarize(root, suffix='', audit_name='frozen_detector.json', activations=('count4', 'relu')):
    root = Path(root)
    report = dict(root=str(root), models={},
        scope='Same-device fixed-weight on/off comparison. Lost GT cohort is outcome-selected; '
              'ROI summaries are descriptive, not causal, and overlapping regions are not independent.')
    for activation in activations:
        prefix = activation + suffix
        diagnostic_path = root / (prefix + '_diagnostic.json')
        diagnostic = json.loads(diagnostic_path.read_text())
        assert diagnostic['passed']
        model = dict(diagnostic=evidence(diagnostic_path, False),
                     gt_transitions70=diagnostic['gt_transitions70'],
                     localized_lost_geometry=diagnostic['localized_lost_geometry'])
        manifests = []
        for mode in ('on', 'off'):
            folder = root / (prefix + '_' + mode)
            ev, replay, index = [evidence(folder / name) for name in
                ('evaluation.json', 'offline_replay.json', 'predictions/index.json')]
            e, r = ev['content'], replay['content']
            assert r['passed'] and r['exact_match'] and r['all_file_hashes_verified']
            assert e['ap'] == r['ap'] and e['ap_3d'] == r['ap_3d']
            assert e['predictions']['index_sha256'] == index['sha256']
            assert len(index['content']['frames']) == e['samples'] == 1738 and e['full_split']
            assert diagnostic['models']['fusion' if mode == 'on' else 'ego']['index_sha256'] == index['sha256']
            manifest_paths = list(folder.glob('manifest_*.json'))
            assert len(manifest_paths) == 1
            manifest = evidence(manifest_paths[0]); manifests.append(manifest['content'])
            model[mode] = dict(evaluation=ev, replay=replay, manifest=manifest,
                              prediction_index=evidence(folder / 'predictions/index.json', False))
            if mode == 'on':
                frames = index['content']['frames']
                by_id = {f['frame_id']: f for f in frames}
                assert len(by_id) == len(frames)
                byte_rows = [dict(frame_id=f['frame_id'], bytes=f['communication']['total_bytes']) for f in frames]
                assert sum(x['bytes'] for x in byte_rows) == e['feature_packets']['total_bytes']
                assert len(byte_rows) == e['feature_packets']['count']
                model['communication_frames'] = byte_rows
                for frame in frames:
                    assert len(frame['fusion_diagnostics']['gt_regions']) == frame['gt_boxes']
                model['feature_statistics'] = dict(
                    frames=aggregate([f['fusion_diagnostics']['frame'] for f in frames]),
                    all_gt=aggregate([g for f in frames for g in f['fusion_diagnostics']['gt_regions']]),
                    lost_gt=aggregate([by_id[x['frame_id']]['fusion_diagnostics']['gt_regions'][x['gt_index']]
                                       for x in diagnostic['lost_examples']]))
        assert manifests[0]['checkpoint_sha256'] == manifests[1]['checkpoint_sha256']
        assert manifests[0]['gpu_visible'] == manifests[1]['gpu_visible']
        # Other research may advance main between sequential jobs. Require the
        # actual inference implementation to match; retain both runtime commits.
        inference_files = ['opencood/models/e3dsnn_fusion.py', 'opencood/models/e3dsnn_single.py',
            'opencood/models/e3dsnn_single_ann.py', 'opencood/models/sub_modules/e3dsnn/backbone_3d.py',
            'opencood/models/sub_modules/e3dsnn/backbone_2d.py', 'opencood/models/fuse_modules/e3dsnn_geometry.py',
            'opencood/tools/train_e3dsnn_single.py', 'opencood/utils/e3dsnn_fusion_probe.py']
        if any('fusion_readout' in manifest for manifest in manifests):
            inference_files.append('opencood/models/fuse_modules/e3dsnn_readout.py')
        blobs = []
        for manifest in manifests:
            blobs.append({path: subprocess.check_output(['git', 'rev-parse',
                manifest['git_commit'] + ':' + path], cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
                for path in inference_files})
        assert blobs[0] == blobs[1], 'On/off inference implementation differs'
        model['verified_inference_blobs'] = blobs[0]
        if suffix:
            training = root / prefix
            for name in ('best_ap70_metrics.json', audit_name, 'status.json'):
                model[name] = evidence(training / name)
            frozen = model[audit_name]['content']
            assert frozen['passed'] and frozen['verified_through_epoch'] == 30
            assert frozen['initial_sha256'] == frozen['current_sha256']
            assert model['status.json']['content']['state'] == 'complete'
            model['best_checkpoint'] = evidence(training / 'best_ap70.pth', False)
            assert model['best_checkpoint']['sha256'] == manifests[0]['checkpoint_sha256']
        report['models'][activation] = model
    report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suffix', default='', help='Run suffix, e.g. _frozen30')
    parser.add_argument('--audit-name', choices=['frozen_detector.json', 'fixed_bn.json'], default='frozen_detector.json')
    parser.add_argument('--activations', nargs='+', choices=['count4', 'relu'], default=['count4', 'relu'])
    args = parser.parse_args()
    result = summarize(args.root, args.suffix, args.audit_name, args.activations)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(passed=result['passed'], output=args.output)), flush=True)
