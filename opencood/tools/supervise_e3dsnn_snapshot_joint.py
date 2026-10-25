"""Replace only a Scheme-2 timeout supervisor, preserving its live trainer.

No model update, optimizer restart, GPU migration or training budget change.
The ready/accepted handshake prevents two finalizers; Linux PID start ticks
and exact command/owner/parent checks protect unrelated processes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
DRIVER='opencood.tools.run_e3dsnn_snapshot_joint'
TRAINER='opencood.tools.train_e3dsnn_single'


def save(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def option(command,key):
    if command.count(key)!=1:raise ValueError('Missing/duplicate option '+key)
    return command[command.index(key)+1]


def validate_commands(parent,child,root,arm):
    if option(parent,'-m')!=DRIVER or option(parent,'--mode')!='coordinate':
        raise ValueError('Not the expected coordinator')
    if option(parent,'--root')!=str(root) or option(parent,'--arm')!=arm:
        raise ValueError('Wrong coordinator experiment')
    if option(child,'-m')!=TRAINER or option(child,'--mode')!='train':
        raise ValueError('Not the expected trainer')
    folder=root/arm
    expected={'--config':str(folder/'train.yaml'),'--output':str(folder/'train'),
        '--checkpoint':str(root/'source.pth'),'--epochs':'10','--selection-iou':'0.7'}
    if any(option(child,k)!=v for k,v in expected.items()) or '--resume' in child:
        raise ValueError('Training command differs from the registered run')


def process(pid):
    p=Path('/proc')/str(pid)
    if p.stat().st_uid!=os.getuid():raise PermissionError('Different process owner')
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    if fields[0]=='Z':raise ProcessLookupError('Process exited')
    return dict(pid=pid,start_ticks=int(fields[19]),parent=int(fields[1]),
        command=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s])


def same(expected):
    try:
        current=process(expected['pid'])
        return current['start_ticks']==expected['start_ticks'] and current['command']==expected['command']
    except (FileNotFoundError,ProcessLookupError):return False


def handoff(root,arm,parent_pid,trainer_pid):
    parent,child=process(parent_pid),process(trainer_pid)
    validate_commands(parent['command'],child['command'],root,arm)
    if child['parent']!=parent_pid:raise ValueError('Trainer is not this coordinator child')
    folder=root/arm
    if (folder/'train/status.json').exists():raise ValueError('Trainer already ended')
    record=folder/'supervision_handoff.json'
    with record.open('x') as f:
        json.dump(dict(parent=parent,trainer=child,root=str(root),arm=arm,
            reason='Observed multi-budget throughput projects beyond original 48h supervisor timeout',
            training_unchanged=True,attention_after_hours=120,created_unix=time.time()),f,indent=2)
    # Only copy runtime controls, never print or serialize the environment.
    entries=(Path('/proc')/str(parent_pid)/'environ').read_bytes().split(b'\0')
    remote_env={k.decode():v.decode() for k,v in (e.split(b'=',1) for e in entries if b'=' in e)}
    env=dict(os.environ)
    for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'OPENCOOD_CUDA_MEMORY_FRACTION','PYTHONPATH'):
        if key not in remote_env:raise ValueError('Missing registered runtime control '+key)
        env[key]=remote_env[key]
    with (folder/'supervision.log').open('x') as log:
        watcher=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--mode','watch',
            '--root',str(root),'--arm',arm],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if (folder/'supervision_ready.json').exists():break
        if watcher.poll() is not None:raise RuntimeError('New supervisor failed before handoff')
        time.sleep(.1)
    else:raise TimeoutError('New supervisor not ready; original remains active')
    if not same(parent) or not same(child):raise RuntimeError('Process changed before handoff')
    os.kill(parent_pid,signal.SIGTERM)  # Exact PID only; never kill the process group.
    for _ in range(100):
        if not same(parent):break
        time.sleep(.1)
    else:raise RuntimeError('Original supervisor still alive; do not finalize twice')
    if not same(child):raise RuntimeError('Trainer did not survive supervisor handoff')
    result=dict(passed=True,watcher_pid=watcher.pid,trainer_pid=trainer_pid,
        old_supervisor_pid=parent_pid,trainer_start_ticks=child['start_ticks'],
        optimizer_and_rng_untouched=True,unix=time.time())
    save(folder/'supervision_accepted.json',result);print(json.dumps(result),flush=True)


def watch(root,arm):
    folder=root/arm;record=read(folder/'supervision_handoff.json');child=record['trainer']
    try:
        if not same(child):raise RuntimeError('Trainer missing before handoff')
        save(folder/'supervision_ready.json',dict(pid=os.getpid(),unix=time.time()))
        for _ in range(600):
            if (folder/'supervision_accepted.json').exists():break
            time.sleep(.1)
        else:raise TimeoutError('Handoff not accepted; no finalizer started')
        started=time.monotonic()
        while same(child):
            if time.monotonic()-started>120*3600:
                save(folder/'supervision_status.json',dict(state='needs_attention',
                    reason='120h supervision window reached; trainer left running, no automatic restart'))
                return
            save(folder/'supervision_status.json',dict(state='watching_existing_trainer',
                trainer_pid=child['pid'],elapsed_seconds=time.monotonic()-started,unix=time.time()))
            time.sleep(60)
        status=read(folder/'train/status.json')
        if status.get('state')!='complete':raise RuntimeError('Trainer failed; no automatic retry')
        save(folder/'supervision_status.json',dict(state='finalizing',unix=time.time()))
        with (folder/'supervised_finish.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m',DRIVER,'--root',str(root),'--arm',arm,
                '--mode','finish'],cwd=str(REPO),stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=120*3600)
        save(folder/'supervision_status.json',dict(state='complete',unix=time.time()))
    except Exception:
        save(folder/'supervision_status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=('fixed2k','multi248k'),required=True)
    p.add_argument('--mode',choices=('handoff','watch'),required=True)
    p.add_argument('--coordinator-pid',type=int);p.add_argument('--trainer-pid',type=int)
    a=p.parse_args()
    if a.mode=='handoff':handoff(a.root,a.arm,a.coordinator_pid,a.trainer_pid)
    else:watch(a.root,a.arm)
