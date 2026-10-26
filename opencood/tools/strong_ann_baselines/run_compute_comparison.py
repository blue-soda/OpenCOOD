"""Execute a registered alternating plan once, keeping failed/interfered runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def await_resource(plan,root,deadline,method,repeat):
    """Queue before measurement; no GPU context or interference with training."""
    if not plan.get('wait_for_empty_gpu'):return True
    successes=0
    while True:
        gpu=str(plan['physical_gpu'])
        raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits']).decode().strip()
        uuid,free,util=[x.strip() for x in raw.split(',')]
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                          '--format=csv,noheader,nounits']).decode().splitlines()
        occupied=[line for line in processes if line.split(',')[0].strip()==uuid]
        ok=not occupied and int(free)>=16000 and int(util)<=20
        row=dict(unix=time.time(),gpu=gpu,uuid=uuid,free_mib=int(free),utilization=int(util),
                 other_compute_processes=occupied,passed=ok,method=method,repeat=repeat)
        with (root/'resource_admission.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        successes=successes+1 if ok else 0
        if successes==5:return True
        if time.time()>=deadline:return False
        write(root/'status.json',dict(state='waiting_for_gpu',method=method,repeat=repeat,last_resource=row,
                                     deadline_unix=deadline,automatic_model_retry=False))
        time.sleep(2 if ok else 300)


def run(plan,root):
    methods=plan['methods'];names=list(methods)
    if len(names)!=4 or plan['repeats']!=3:raise ValueError('Registered four methods / three repeats only')
    rows=[];failed=set();deadline=time.time()+plan.get('resource_wait_seconds',21600)
    for repeat in range(3):
        order=names[repeat:]+names[:repeat]
        for name in order:
            if name in failed:continue
            if not await_resource(plan,root,deadline,name,repeat):
                write(root/'status.json',dict(state='resource_timeout',method=name,repeat=repeat,completed_runs=len(rows)))
                return
            spec=methods[name];output=root/('%s_repeat%d'%(name,repeat))
            if output.exists():raise ValueError('Refuse to overwrite or automatically retry '+str(output))
            cmd=spec['command']+['--output',str(output),'--compute-count','64','--compute-repeat',str(repeat)]
            write(root/'status.json',dict(state='running',method=name,repeat=repeat,completed=len(rows),command=cmd))
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            started=time.time()
            with (root/(output.name+'.log')).open('wb') as f:
                # GNU timeout terminates the child's process group; the supervisor
                # never retries failures or terminates unrelated GPU processes.
                process=subprocess.Popen(['timeout','--kill-after=30',str(plan['per_run_timeout_seconds'])]+cmd,
                    stdout=f,stderr=subprocess.STDOUT,cwd=spec['cwd'],env=env,start_new_session=True)
                write(root/'active_child.json',dict(pid=process.pid,method=name,repeat=repeat,started_unix=started))
                code=process.wait()
            summary=output/'compute_summary.json';result=None
            if code==0 and summary.exists():
                result=json.loads(summary.read_text())
                if not result['passed'] or result['unique_frames']!=64 or result['timed_calls']!=128:code=90
            elif code==0:code=91
            row=dict(method=name,repeat=repeat,returncode=code,started_unix=started,finished_unix=time.time(),
                     output=str(output),eligible_for_speed_table=spec['eligible_for_speed_table'])
            if result is not None:
                row['summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
                row['scope_statistics']=[dict(scope=x['scope'],seconds=x['seconds'],
                    peak_allocated_bytes=x['peak_allocated_bytes'],peak_reserved_bytes=x['peak_reserved_bytes']) for x in result['results']]
            if code!=0:failed.add(name)
            rows.append(row);write(root/'results_so_far.json',rows)
    write(root/'summary.json',dict(completed=len(rows)==12 and not failed,rows=rows,
          failed_methods=sorted(failed),automatic_retries=False,outliers_removed=False,
          note='CoSDH reference retains redundant audit work; not eligible for fair speed ranking. Inspect raw interference before claims.'))
    write(root/'status.json',dict(state='complete' if len(rows)==12 and not failed else 'partial_failure',
          completed_runs=len(rows),failed_methods=sorted(failed)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(a.plan).read_text());write(root/'plan.json',plan)
    try:run(plan,root)
    except Exception:
        write(root/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
