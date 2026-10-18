import sys,json,subprocess,time,os,signal,traceback
from pathlib import Path
plan_path=Path(sys.argv[1]); plan=json.loads(plan_path.read_text()); root=plan_path.parent
status=root/(plan['name']+'_pipeline.json')
def update(value):
    value.update(supervisor_pid=os.getpid(),updated_unix=time.time())
    temp=status.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)); temp.replace(status)
try:
    for job in plan['jobs']:
        with (root/(job['name']+'.log')).open('wb') as log:
            child=subprocess.Popen(job['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(dict(state='running',job=job['name'],child_pid=child.pid,command=job['command']))
            try:
                code=child.wait(timeout=job['timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                raise RuntimeError('Hard timeout: '+job['name'])
            if code: raise RuntimeError('Exit '+str(code)+': '+job['name'])
    update(dict(state='complete',jobs=[j['name'] for j in plan['jobs']]))
except Exception:
    update(dict(state='failed',error=traceback.format_exc())); raise
