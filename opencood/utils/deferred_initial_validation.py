"""Run fixed epoch-zero evaluation alongside training, retaining it in best selection."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import torch


def launch(args, output):
    if not args.checkpoint:
        raise ValueError('Deferred initial evaluation requires a fixed checkpoint')
    root = Path(output) / 'initial_eval'
    log = (Path(output) / 'initial_eval.log').open('xb')
    env = dict(os.environ, OPENCOOD_CUDA_MEMORY_FRACTION='.2')
    command = [sys.executable, '-u', '-m', 'opencood.tools.train_e3dsnn_single',
        '--mode', 'eval', '--config', args.config, '--checkpoint', args.checkpoint,
        '--output', str(root), '--workers', str(args.workers), '--seed', str(args.seed)]
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    def cleanup():
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
        log.close()
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    record = dict(pid=child.pid, command=command, started_unix=time.time(),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        semantics='Same fixed epoch-zero checkpoint evaluated concurrently; included before first best selection')
    (Path(output) / 'initial_eval_launch.json').write_text(json.dumps(record, indent=2))
    # A sequential validation DataLoader consumes one parent CPU base-seed draw.
    # Workers own their NumPy/Python randomness; the model is deterministic in eval.
    torch.empty((), dtype=torch.int64).random_()
    return child, root, record


def collect(handle, checkpoint, expected_manifest):
    child, root, record = handle
    code = child.wait(timeout=max(1, 86400 - (time.time() - record['started_unix'])))
    if code:
        raise RuntimeError('Deferred epoch-zero validation failed: {}'.format(code))
    status = json.loads((root / 'status.json').read_text())
    manifests = list(root.glob('manifest_*.json'))
    assert status['state'] == 'complete' and len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    for key in ('checkpoint_sha256', 'config_sha256', 'data_manifest_sha256', 'val_samples'):
        assert manifest[key] == expected_manifest[key], key
    assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == record['checkpoint_sha256']
    metrics = json.loads((root / 'evaluation.json').read_text())
    assert metrics['full_split'] and metrics['samples'] == expected_manifest['val_samples']
    metrics['epoch'] = 0
    state = torch.load(checkpoint, map_location='cpu')
    return SimpleNamespace(state_dict=lambda: state), metrics
