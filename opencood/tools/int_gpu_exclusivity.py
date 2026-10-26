"""Fail this benchmark on sampled GPU contention; never control other jobs."""
import os
import subprocess
import time
from pathlib import Path


class GPUContention(RuntimeError):
    def __init__(self,index,foreign):
        super().__init__('GPU {} contention with PIDs {}; discard timing and stop only this benchmark'.format(index,foreign))
        self.pids=foreign;self.evidence=[]


def process_identity(pid):
    # Do not collect arbitrary command-line arguments, which may contain secrets.
    result=dict(pid=pid)
    try:
        lines=Path('/proc/{}/status'.format(pid)).read_text().splitlines()
        result.update({line.split(':',1)[0]:line.split(':',1)[1].strip() for line in lines
                       if line.split(':',1)[0] in ('Name','PPid','Uid')})
    except OSError:result['unavailable']=True
    return result


def verify_gpu_processes(gpu_rows,process_rows,index,pid,allow_shared=False):
    devices={}
    for row in gpu_rows.splitlines():
        if not row.strip():continue
        fields=[v.strip() for v in row.split(',')]
        if len(fields)!=2 or not fields[0].isdigit() or not fields[1].startswith('GPU-'):
            raise RuntimeError('GPU device query unavailable or malformed: '+row)
        devices[int(fields[0])]=fields[1]
    if index not in devices:raise RuntimeError('Physical benchmark GPU not found')
    uuid=devices[index];pids=[]
    for row in process_rows.splitlines():
        if not row.strip():continue
        fields=[v.strip() for v in row.split(',')]
        if len(fields)!=2 or not fields[0].startswith('GPU-') or not fields[1].isdigit():
            raise RuntimeError('GPU process query unavailable or malformed: '+row)
        if fields[0]==uuid:pids.append(int(fields[1]))
    if pid not in pids:raise RuntimeError('Benchmark PID absent from GPU query; cannot validate isolation')
    foreign=sorted(set(pids)-{pid})
    if foreign and not allow_shared:raise GPUContention(index,foreign)
    result=dict(gpu=index,uuid=uuid,pids=pids)
    if allow_shared:result.update(foreign_pids=foreign,timing_valid=not foreign)
    return result


def sample_gpu_usage(index):
    """Observe shared training/validation without waiting or controlling jobs.

    Sharing affects latency comparability, not permission to compute. Hardware
    and malformed-query errors still propagate; CUDA handles actual allocation.
    """
    def query(fields):
        return subprocess.check_output(['nvidia-smi','-i',str(index),fields,
            '--format=csv,noheader,nounits'],timeout=15).decode()
    result=verify_gpu_processes(query('--query-gpu=index,uuid'),
        query('--query-compute-apps=gpu_uuid,pid'),index,os.getpid(),allow_shared=True)
    result['checked_unix']=time.time()
    return result


def check_gpu_exclusive(index):
    def query(fields):
        # Scope both queries to the physical task GPU. An unrelated failed GPU
        # can otherwise inject diagnostic text into the all-device CSV output.
        return subprocess.check_output(['nvidia-smi','-i',str(index),fields,'--format=csv,noheader,nounits'],timeout=15).decode()
    def snapshot():
        return verify_gpu_processes(query('--query-gpu=index,uuid'),
            query('--query-compute-apps=gpu_uuid,pid'),index,os.getpid())
    try:
        result=snapshot();result['timing_valid']=True
    except GPUContention as first:
        evidence=[dict(checked_unix=time.time(),processes=[process_identity(p) for p in first.pids])]
        # No model kernels run during this bounded recheck. A transient event
        # invalidates the WHOLE evaluation's latency, never silently ignored.
        time.sleep(1.)
        try:result=snapshot()
        except GPUContention as second:
            evidence.append(dict(checked_unix=time.time(),processes=[process_identity(p) for p in second.pids]))
            second.evidence=evidence;raise
        result.update(timing_valid=False,transient_contention=evidence)
    result['checked_unix']=time.time()
    return result


def wait_for_training_gpu(index,timeout_seconds=600,poll_seconds=15,on_event=None):
    """Cooperatively pause kernels at a training boundary, never kill other jobs.

    Allocated model/state memory remains resident. Hardware/query failures are
    not contention and still fail closed. Benchmark evaluation stays strict.
    """
    if timeout_seconds<=0 or poll_seconds<=0:raise ValueError('Wait limits must be positive')
    start=time.monotonic();events=[]
    while True:
        try:result=check_gpu_exclusive(index)
        except GPUContention as exc:
            events.extend(exc.evidence)
            remaining=timeout_seconds-(time.monotonic()-start)
            if on_event:on_event(dict(status='WAIT_TIMEOUT' if remaining<=0 else 'WAITING_GPU',
                evidence=events,pids=exc.pids,remaining_seconds=max(remaining,0.)))
            if remaining<=0:raise
            time.sleep(min(poll_seconds,remaining))
        else:
            if events:
                result.update(timing_valid=False,training_wait_seconds=time.monotonic()-start,training_contention=events)
                if on_event:on_event(dict(status='RESUMED',evidence=events,result=result))
            return result
