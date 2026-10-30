"""Execute an immutable, explicit research job list without automatic retries.

Operational supervisor only: model, training and evaluation remain in their
existing entry points. Each job has a fresh log and a bounded process group.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def coordinate(protocol_path):
    root = protocol_path.parent
    protocol = json.loads(protocol_path.read_text())
    repo = Path(protocol['repo'])
    for path, expected in protocol['input_hashes'].items():
        assert sha(path) == expected, path
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip()
    assert head == protocol['code_commit'], (head, protocol['code_commit'])
    try:
        for job in protocol['jobs']:
            write(root / 'status.json', dict(state='running', stage=job['name'], unix=time.time()))
            command = [sys.executable if a == '{python}' else a for a in job['command']]
            with (root / (job['name'] + '.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=str(repo), env=os.environ.copy(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                write(root / 'child.json', dict(pid=child.pid, command=command, stage=job['name']))
                try:
                    code = child.wait(timeout=job['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, command)
            for output in job.get('required_outputs', []):
                assert Path(output).is_file(), output
        write(root / 'status.json', dict(state='complete', unix=time.time(),
            protocol_sha256=sha(protocol_path), code_commit=head))
    except Exception:
        write(root / 'status.json', dict(state='failed', unix=time.time(),
            error=traceback.format_exc(), automatic_retry=False))
        raise


def launch(protocol_path, gpu):
    root = protocol_path.parent
    protocol = json.loads(protocol_path.read_text())
    # Exclusive marker disallows duplicate launches, including after failure.
    with (root / 'launch.json').open('x') as f:
        json.dump(dict(state='admitting', unix=time.time()), f)
    readings = []
    for _ in range(3):
        raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode()
        free, util = map(int, raw.strip().split(','))
        assert free >= 16000 and util <= 20, raw
        readings.append(dict(free_mib=free, utilization=util))
        time.sleep(2)
    assert os.statvfs(str(root)).f_bavail * os.statvfs(str(root)).f_frsize >= 5 * 1024**3
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=protocol['repo'],
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMPY_MADVISE_HUGEPAGE='0', OPENCOOD_CUDA_MEMORY_FRACTION='.6')
    # Short IPC path avoids Unix-domain socket path limits with DataLoader.
    tmp = Path('/tmp') / ('cpsnn_' + str(os.getpid()))
    tmp.mkdir()
    env['TMPDIR'] = str(tmp)
    subprocess.run([sys.executable, '-c',
        'import torch; x=torch.ones(32).cuda(); assert x.sum().item()==32; torch.cuda.synchronize()'],
        env=env, check=True, timeout=120)
    with (root / 'supervisor.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name,
            '--protocol', str(protocol_path), '--mode', 'run'], cwd=protocol['repo'],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    record = dict(pid=child.pid, gpu=gpu, readings=readings, unix=time.time(),
                  protocol_sha256=sha(protocol_path), tmpdir=str(tmp))
    write(root / 'launch.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--mode', choices=('launch', 'run'), required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    path = Path(args.protocol).resolve()
    if args.mode == 'launch':
        if args.gpu is None:
            parser.error('--gpu is required for launch')
        launch(path, args.gpu)
    else:
        coordinate(path)
