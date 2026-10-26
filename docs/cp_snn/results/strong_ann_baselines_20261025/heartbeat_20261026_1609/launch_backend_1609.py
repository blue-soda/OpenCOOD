import os,json,time,pathlib,subprocess,hashlib
b=pathlib.Path('/data2/tzm/cp_snn_recovery_20261026')
r=b/'scheme2_backend_native01';r.mkdir(exist_ok=False)
source=b/'compute64_source04';gpu='6'
def write(name,v):
 p=r/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
spec=json.loads((b/'compute64_suite01/plan.json').read_text())['methods']['scheme2_2k']
command=spec['command'];command[command.index('--gpu')+1]=gpu
plan=dict(source_commit='387677b',source_archive_sha256=hashlib.sha256((b/'compute64_source04.tar').read_bytes()).hexdigest(),
          source=str(source),gpu=gpu,counts=[4,64,64],timeout_seconds=1500,
          backend='native_deterministic',model_update=False,automatic_retry=False,command=command)
write('plan.json',plan)
deadline=time.time()+7200;rows=[]
for i,count in enumerate(plan['counts']):
 good=0
 while good<5:
  raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
  uuid,free,util=[x.strip() for x in raw.split(',')]
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).decode().splitlines()
  own=[x for x in apps if x.split(',')[0].strip()==uuid]
  ok=not own and int(free)>=16000 and int(util)<=20
  row=dict(unix=time.time(),gpu=gpu,free_mib=int(free),utilization=int(util),processes=own,ok=ok)
  with (r/'resource.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  good=good+1 if ok else 0
  if good==5:break
  if time.time()>deadline:write('status.json',dict(state='resource_timeout',rows=rows));raise SystemExit(0)
  write('status.json',dict(state='waiting_for_gpu',run=i,count=count))
  time.sleep(2 if ok else 300)
 out=r/('run%d_count%d'%(i,count))
 cmd=command+['--output',str(out),'--compute-count',str(count),'--diagnose','--backend','native_deterministic']
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (r/('run%d.log'%i)).open('wb') as log:
  p=subprocess.Popen(['timeout','--kill-after=30','1500']+cmd,cwd=str(source),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write('status.json',dict(state='running',run=i,count=count,pid=p.pid));code=p.wait()
 rows.append(dict(run=i,count=count,returncode=code,output=str(out)))
 write('runs.json',rows)
 if code:write('status.json',dict(state='failed',run=i,returncode=code,automatic_retry=False));raise SystemExit(0)
summaries=[json.loads((pathlib.Path(x['output'])/'diagnostic_summary.json').read_text()) for x in rows]
maps=[{x['frame_id']:x['actual_fingerprint'] for x in s['rows'] if x['repeat']==0} for s in summaries]
cross=[]
for i,j in [(0,1),(1,2),(0,2)]:
 common=sorted(set(maps[i])&set(maps[j]));bad=[f for f in common if maps[i][f]!=maps[j][f]]
 cross.append(dict(left=i,right=j,common=len(common),different_frames=bad))
result=dict(diagnostic_only=True,no_timing_or_AP_claim=True,cross_process=cross,
 within_process_unstable=[s['unstable_repeat_calls'] for s in summaries],
 historical_mismatch_calls=[s['mismatched_calls'] for s in summaries],
 stable_on_sample=all(not x['different_frames'] for x in cross) and all(s['unstable_repeat_calls']==0 for s in summaries))
write('summary.json',result);write('status.json',dict(state='complete',stable_on_sample=result['stable_on_sample']))
