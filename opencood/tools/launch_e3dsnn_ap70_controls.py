"""Start one prepared AP70 control pipeline only on an unoccupied GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(root, activation, gpu):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[2]
    for name in ('count4', 'relu'):
        if not json.loads((root / (name + '_diagnostic.json')).read_text()).get('passed'):
            raise ValueError('Complete both CPU diagnostics first')
    plan_path = root / (activation + '_controls_plan.json')
    plan = json.loads(plan_path.read_text())
    if plan['name'] != activation + '_controls':
        raise ValueError('Plan name mismatch')
    status = root / (plan['name'] + '_pipeline.json')
    lock = root / (plan['name'] + '_launch.json')
    if status.exists() or lock.exists():
        raise FileExistsError('Pipeline already attempted; inspect status, never duplicate/retry automatically')
    for other in root.glob('*_controls_launch.json'):
        previous = json.loads(other.read_text())
        previous_status = other.with_name(other.name.replace('_launch.json', '_pipeline.json'))
        state = json.loads(previous_status.read_text())['state'] if previous_status.exists() else 'launching'
        if previous.get('gpu') == gpu and state not in ('complete', 'failed'):
            print(json.dumps(dict(state='waiting_for_free_gpu', gpu=gpu, reserved_by=other.name)))
            return False
    # Prepared plans contain the full frozen fusion initializer and single initializer.
    for job, flag, digest in [(plan['jobs'][0], '--checkpoint', plan['fusion_checkpoint_sha256']),
                              (plan['jobs'][2], '--single-checkpoint', plan['single_checkpoint_sha256'])]:
        checkpoint = Path(job['command'][job['command'].index(flag) + 1])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
            raise ValueError('Initializer hash mismatch')
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).decode().strip()
    used, free, utilization = [int(x.strip()) for x in row.split(',')]
    if used > 1000 or free < 12000 or utilization > 10:
        print(json.dumps(dict(state='waiting_for_free_gpu', gpu=gpu, memory_used_mib=used,
                              memory_free_mib=free, utilization_percent=utilization)))
        return False
    metadata = dict(gpu=gpu, started_unix=time.time(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo)).decode().strip())
    # Exclusive creation prevents a second monitor from launching this same pipeline.
    with lock.open('x') as stream:
        json.dump(dict(metadata, state='launching'), stream, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(repo),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMPY_MADVISE_HUGEPAGE='0')
    with (root / (plan['name'] + '_supervisor.log')).open('xb') as log:
        process = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(plan_path)],
            cwd=str(repo), env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    metadata.update(state='launched', supervisor_pid=process.pid)
    lock.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--activation', required=True, choices=['count4', 'relu'])
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    launch(args.root, args.activation, args.gpu)
