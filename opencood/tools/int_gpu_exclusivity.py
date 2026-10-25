"""Fail this benchmark on sampled GPU contention; never control other jobs."""
import os
import subprocess
import time


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
    if foreign:raise RuntimeError('GPU {} contention with PIDs {}; discard timing and stop only this benchmark'.format(index,foreign))
    return dict(gpu=index,uuid=uuid,pids=pids)


def check_gpu_exclusive(index):
    def query(fields):
        return subprocess.check_output(['nvidia-smi',fields,'--format=csv,noheader,nounits'],timeout=15).decode()
    result=verify_gpu_processes(query('--query-gpu=index,uuid'),
        query('--query-compute-apps=gpu_uuid,pid'),index,os.getpid())
    result['checked_unix']=time.time()
    return result
