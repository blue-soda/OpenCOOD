"""Verify matched GRU/spiking-GRU pilots and exact spiking checkpoint restart."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from opencood.data_utils.int_sequence_order import segment_order
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.tools.audit_int_memory_pilot import exact, lines, read, require
from opencood.tools.launch_int_memory import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--resume-gpu', required=True, type=int)
    a = p.parse_args()
    root = Path(a.root).resolve()
    require(not (root/'engineering_gate.json').exists(), 'Gate already exists; inspect previous evidence')
    launch = read(root/'launch_pilot_gru_spike_gru.json')
    manifest = read(root/'pilot_manifest_gru_spike_gru.json')
    command = launch['jobs'][0]['command']
    config_path = command[command.index('--config')+1]
    checkpoint = command[command.index('--checkpoint')+1]
    require(digest(config_path) == launch['config_sha256'] and
            digest(checkpoint) == launch['checkpoint_sha256'], 'Pilot input changed')
    config = load_yaml(config_path)
    initial = torch.load(checkpoint, map_location='cpu')
    report = dict(status='RUNNING', config_sha256=digest(config_path),
                  checkpoint_sha256=digest(checkpoint),
                  parent_manifest_sha256=launch['parent_manifest_sha256'],
                  scope='Engineering coverage/learning/restart, not scientific AP', modes={})
    initialized = {}
    for mode in ('gru', 'spike_gru'):
        torch.manual_seed(303)
        model = PointPillarINT(config['model']['args'], mode)
        model.load_single_checkpoint(initial)
        require(sum(p.numel() for p in model.parameters()) == 30776212, 'Full parameter budget')
        require(sum(p.numel() for p in model.feature_memory.parameters()) == 259712, 'FM parameter budget')
        initialized[mode] = model.state_dict()
    exact(initialized['gru'], initialized['spike_gru'], 'initial_model')
    for mode in ('gru', 'spike_gru'):
        directory = root/('pilot_'+mode)
        require(not (directory/'failure.json').exists(), 'Failure evidence: '+mode)
        result = read(directory/'result.json')
        require(result['status'] == 'EPOCH_LIMIT_REACHED' and result['epochs'] == 2, 'Pilot incomplete')
        contract = read(directory/'manifest.json')['contract']
        require(contract['config_sha256'] == report['config_sha256'] and
                contract['initialization_sha256'] == report['checkpoint_sha256'] and
                contract['manifest_sha256'] == digest(root/'pilot_manifest_gru_spike_gru.json') and
                contract['tbptt_steps'] == 4 and contract['fm_only_epochs'] == 1, 'Pilot contract mismatch')
        values, previous = [], initialized[mode]
        for epoch in (1, 2):
            folder = directory/('epoch_%03d' % epoch)
            frames, evaluation = lines(folder/'train_frames.jsonl'), lines(folder/'eval_frames.jsonl')
            require([r['frame'] for r in frames] == [r['frame'] for r in segment_order(manifest['train'],303+epoch)], 'Training sequence')
            require([r['frame'] for r in evaluation] == [r['frame'] for r in manifest['val']], 'Validation sequence')
            summary = read(folder/'train_summary.json')
            require(summary['supervised_frames'] == sum(r['supervised'] for r in manifest['train']) and
                    summary['updates'] > 0 and math.isfinite(summary['loss_mean']), 'Supervision/learning')
            gradients = [r['temporal_gradient_norm'] for r in frames if r.get('optimizer_step')]
            require(gradients and all(math.isfinite(x) for x in gradients) and max(gradients)>0, 'FM gradient')
            current = torch.load(str(folder/'trained.pth'), map_location='cpu')['model']
            spatial_changed = any(not torch.equal(current[k], initial[k]) for k in initial)
            require(spatial_changed == (epoch == 2), 'Spatial freeze/learning')
            require(any(not torch.equal(current[k], previous[k]) for k in current
                        if k.startswith('feature_memory.')), 'FM parameters did not change')
            require(all(r['state_bytes'] == 26208000 for r in frames+evaluation), 'Cache budget')
            info = dict(epoch=epoch, training=summary, evaluation=read(folder/'eval_summary.json'),
                        spatial_changed=spatial_changed, temporal_gradient_max=max(gradients))
            if mode == 'spike_gru':
                for phase, rows in (('train',frames), ('val',evaluation)):
                    rate = sum(r['emission_nonzero_fraction'] for r in rows)/len(rows)
                    require(0 < rate < 1 and all(math.isfinite(r['membrane_abs_max']) for r in rows), 'Degenerate events/state')
                    info[phase+'_events'] = dict(nonzero_mean=rate,
                        positive_mean=sum(r['emission_positive_fraction'] for r in rows)/len(rows),
                        negative_mean=sum(r['emission_negative_fraction'] for r in rows)/len(rows))
            values.append(info); previous = current
        report['modes'][mode] = values
    del initialized, initial, model, current, previous
    (root/'pilot_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    memory, util = subprocess.check_output(['nvidia-smi','-i',str(a.resume_gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    require(int(memory)>=18000 and int(util)<=5, 'Restart GPU is not idle')
    command = list(next(job for job in launch['jobs'] if job['mode']=='spike_gru')['command'])
    target = root/'pilot_spike_gru_restart'
    require(not target.exists(), 'Preserve previous restart evidence')
    command[command.index('--output')+1] = str(target)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.resume_gpu), PYTHONPATH=str(Path.cwd()),
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0')
    with (root/'pilot_spike_gru_restart.log').open('x') as log:
        subprocess.run(command+['--stop-after-epoch','1'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        require(read(target/'result.json')['status']=='CHECKPOINT_BOUNDARY_STOP', 'Restart boundary')
        subprocess.run(command+['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    reference = torch.load(str(root/'pilot_spike_gru/latest.pth'),map_location='cpu')
    resumed = torch.load(str(target/'latest.pth'),map_location='cpu')
    for key in ('model','optimizer','scheduler','contract','epoch','best','significant_best','stale'):
        exact(reference[key],resumed[key],key)
    for original, repeated in zip(reference['history'],resumed['history']):
        exact(original['evaluation']['metrics'],repeated['evaluation']['metrics'],'restarted_AP')
    report.update(status='PASSED',restart='Exact model/optimizer/scheduler/contract/AP',
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())
    (root/'engineering_gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],restart=report['restart'],output=str(root/'engineering_gate.json'))))


if __name__ == '__main__':
    main()
