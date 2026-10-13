"""Audit completed INT memory comparisons and export a descriptive stage report."""
import argparse
import json
import math
from pathlib import Path

import torch

from opencood.tools.launch_int_memory import digest


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    require(not output.exists(), 'Report output exists; preserve previous evidence')
    report = dict(status='AUDITED', scope='Single-seed validation-selected stage 1; BEV AP, not 3D AP',
                  modes={}, limitations=[
                      'No independent test split or multi-seed uncertainty estimate.',
                      'Reset/no-align are fixed-weight inference interventions, not retrained models.',
                      'LIF/leaky have equal parameters and state width; GRU differs in architecture and parameter count.',
                      'Diagnostic timings on separate GPUs are not a controlled latency/energy benchmark.',
                      'The leaky control jointly removes binary quantization and reset; their effects are not separated.'])
    curves = {}
    for mode in ('lif', 'leaky', 'gru'):
        train, replay = root/('full_'+mode), root/('replay_'+mode)
        result, curve, best = read(train/'result.json'), read(train/'curve.json'), read(train/'best.json')
        runtime = read(train/'effective_runtime.json')['arguments']
        manifest = read(runtime['manifest'])
        contract = read(train/'manifest.json')['contract']
        require(digest(runtime['manifest']) == contract['manifest_sha256'], 'Manifest changed')
        require(digest(runtime['config']) == contract['config_sha256'], 'Configuration changed')
        require(digest(best['checkpoint']) == best['sha256'], 'Best checkpoint changed')
        require(result['status'] == 'PLATEAU_REACHED', 'Training not at validation plateau')
        saved = torch.load(str(train/'latest.pth'), map_location='cpu')
        require(saved['epoch'] == len(curve) and saved['epoch'] >= contract['min_epochs'] and
                saved['stale'] >= contract['patience'] and
                saved['optimizer']['param_groups'][0]['lr'] <= contract['lr']/8, 'Stop rule not satisfied')
        del saved
        for epoch in curve:
            t, v = epoch['training'], epoch['evaluation']
            require(t['scans'] == len(manifest['train']) and
                    t['supervised_frames'] == sum(r['supervised'] for r in manifest['train']), 'Training coverage')
            require(v['scans'] == len(manifest['val']) and
                    v['labels'] == sum(r['supervised'] for r in manifest['val']), 'Validation coverage')
            require(math.isfinite(t['loss_mean']) and all(math.isfinite(x) for x in v['metrics'].values()), 'Nonfinite epoch')
        require(read(replay/'status.json')['status'] == 'COMPLETED', 'Replay worker incomplete')
        selected = next(r for r in curve if r['epoch'] == best['epoch'])
        values, audits = {}, {}
        segments = 1 + sum(a['segment'] != b['segment'] for a, b in zip(manifest['val'], manifest['val'][1:]))
        for policy in ('aligned', 'reset', 'no-align'):
            directory = replay/policy
            evidence = read(directory/'result.json')
            provenance = read(directory/'manifest.json')
            require(evidence['status'] == 'COMPLETED', 'Evaluation not completed')
            require(provenance['checkpoint_sha256'] == best['sha256'] and
                    provenance['manifest_sha256'] == contract['manifest_sha256'], 'Provenance mismatch')
            frames = [json.loads(line) for line in (directory/'eval_frames.jsonl').read_text().splitlines()]
            require(len(frames) == len(manifest['val']), 'Evaluation frame count')
            for frame, expected in zip(frames, manifest['val']):
                require(not frame.get('invalid') and frame['frame'] == expected['frame'] and
                        frame['scene'] == expected['segment'] and frame['timestamp_us'] == expected['timestamp_us'] and
                        frame['supervised'] == expected['supervised'], 'Evaluation sequence mismatch')
            summary = evidence['evaluation']
            require(summary['labels'] == 1738 and summary['gt_boxes'] == 22971, 'Label/GT coverage')
            warps = sum(bool(r['warp']) for r in frames)
            resets = sum(bool(r['reset']) for r in frames)
            require(warps == (len(frames)-segments if policy == 'aligned' else 0), 'Warp policy mismatch')
            require(resets == (len(frames) if policy == 'reset' else segments), 'Reset policy mismatch')
            values[policy] = summary
            audits[policy] = dict(warp_frames=warps, reset_frames=resets, frames=len(frames))
            if mode != 'gru':
                audits[policy].update(emission_nonzero_mean=sum(r['emission_nonzero_fraction'] for r in frames)/len(frames),
                                     emission_on_observed_mean=sum(r['emission_on_observed_fraction'] for r in frames)/len(frames))
        error = max(abs(values['aligned']['metrics'][key]-value)
                    for key, value in selected['evaluation']['metrics'].items())
        require(error <= 1e-6, 'Independent replay AP mismatch')
        ap = {policy: value['metrics']['bev_ap70'] for policy, value in values.items()}
        report['modes'][mode] = dict(epochs=len(curve), best=best, replay=values, audits=audits,
                                    max_replay_ap_error=error,
                                    aligned_minus_reset_pp=100*(ap['aligned']-ap['reset']),
                                    aligned_minus_noalign_pp=100*(ap['aligned']-ap['no-align']))
        curves[mode] = curve
    score = lambda mode, policy: report['modes'][mode]['replay'][policy]['metrics']['bev_ap70']
    report['comparisons_pp'] = dict(lif_minus_leaky=100*(score('lif','aligned')-score('leaky','aligned')),
                                    lif_minus_gru=100*(score('lif','aligned')-score('gru','aligned')),
                                    lif_minus_leaky_after_reset=100*(score('lif','reset')-score('leaky','reset')))
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, curve in curves.items():
        epochs = [r['epoch'] for r in curve]
        axes[0].plot(epochs, [r['training']['loss_mean'] for r in curve], label=mode.upper())
        axes[1].plot(epochs, [100*r['evaluation']['metrics']['bev_ap70'] for r in curve], label=mode.upper())
        best = report['modes'][mode]['best']
        axes[1].scatter([best['epoch']], [100*best['bev_ap70']], s=24)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel('Mean labelled detection loss')
    axes[1].set_ylabel('Validation BEV AP70 (%)')
    fig.suptitle('INT memory stage 1: seed 303, real-frame TBPTT=4')
    fig.tight_layout()
    fig.savefig(str(output/'training_curves.png'), dpi=180)
    fig.savefig(str(output/'training_curves.svg'))
    plt.close(fig)
    print(json.dumps(dict(status=report['status'], comparisons_pp=report['comparisons_pp'], output=str(output))))


if __name__ == '__main__':
    main()
