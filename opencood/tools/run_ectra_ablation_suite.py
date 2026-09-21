"""Run isolated, fixed-weight ECTRA interventions with logged timeouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
from opencood.tools.ectra_ablation_utils import ABLATIONS, DEFAULT_ABLATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--modes', default=','.join(DEFAULT_ABLATIONS))
    parser.add_argument('--limit', type=int, default=256,
                        help='0 means full evaluation; screening AP is not final AP')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--p', type=float, default=0.3)
    args = parser.parse_args()
    modes = args.modes.split(',')
    if len(set(modes)) != len(modes) or any(m not in ABLATIONS for m in modes):
        parser.error('modes must be unique members of %s' % (ABLATIONS,))
    if args.limit < 0 or args.timeout <= 0:
        parser.error('limit >= 0 and timeout > 0 required')
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    if not checkpoint.is_file() or not config.is_file():
        parser.error('checkpoint and config must be existing files')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sha = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    manifest = dict(vars(args), checkpoint_sha256=sha.hexdigest(),
                    screening=args.limit > 0)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['PYTHONPATH'] = os.getcwd()
    results = []
    for mode in modes:
        folder = output / mode
        folder.mkdir()
        shutil.copy2(config, folder / 'config.yaml')
        try:
            os.link(str(checkpoint), str(folder / checkpoint.name))
        except OSError:
            shutil.copy2(checkpoint, folder / checkpoint.name)
        command = [sys.executable, '-u', 'opencood/tools/inference.py',
                   '--model_dir', str(folder), '--fusion_method', 'intermediate',
                   '--two_stage', '1', '--dataset', 'd', '--save_vis_interval', '0',
                   '--p', str(args.p), '--eval_seed', str(args.seed),
                   '--debug_max_samples', str(args.limit), '--note', mode,
                   '--ectra_ablation', mode, '--ectra_diagnostics',
                   str(folder / 'diagnostics.jsonl')]
        start = time.time()
        row = {'mode': mode, 'command': command, 'started_at': start}
        with (folder / 'inference.log').open('wb') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
            row['pid'] = process.pid
            (output / 'active.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            try:
                row['returncode'] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                row['returncode'] = process.returncode
                row['timeout'] = True
        row['seconds'] = time.time() - start
        log = (folder / 'inference.log').read_text(errors='replace')
        matches = re.findall(r'Average Precision at IOU (0\.[357]) is ([0-9.]+)', log)
        row['ap'] = dict(matches)
        diagnostics = folder / 'diagnostics.jsonl'
        if diagnostics.exists():
            records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
            identifiers = [(record['sample_idx'], record['time_intervals'])
                           for record in records]
            if not records or any(index is None or times is None
                                  for index, times in identifiers):
                raise RuntimeError('Missing sample/time identity in diagnostics')
            row['sample_count'] = len(records)
            row['sample_order_sha256'] = hashlib.sha256(
                json.dumps(identifiers).encode()).hexdigest()
            if results and row['sample_order_sha256'] != results[0]['sample_order_sha256']:
                row['pairing_error'] = True
        results.append(row)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        if row['returncode'] or len(matches) != 3 or row.get('pairing_error'):
            raise RuntimeError('Ablation failed; inspect %s' % folder)
    (output / 'active.json').write_text(json.dumps({'status': 'complete'}))


if __name__ == '__main__':
    main()
