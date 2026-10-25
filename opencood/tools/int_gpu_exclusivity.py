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


def verify_gpu_processes(gpu_rows,process_rows,index,pid):
    devices={int(row.split(',')[0].strip()):row.split(',')[1].strip()
             for row in gpu_rows.splitlines() if row.strip()}
    if index not in devices:raise RuntimeError('Physical benchmark GPU not found')
    uuid=devices[index];pids=[]
    for row in process_rows.splitlines():
        if not row.strip():continue
        fields=[v.strip() for v in row.split(',')]
        if fields[0]==uuid:pids.append(int(fields[1]))
    if pid not in pids:raise RuntimeError('Benchmark PID absent from GPU query; cannot validate isolation')
    foreign=sorted(set(pids)-{pid})
    if foreign:raise GPUContention(index,foreign)
    return dict(gpu=index,uuid=uuid,pids=pids)


def check_gpu_exclusive(index):
    def query(fields):
        return subprocess.check_output(['nvidia-smi',fields,'--format=csv,noheader,nounits'],timeout=15).decode()
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
