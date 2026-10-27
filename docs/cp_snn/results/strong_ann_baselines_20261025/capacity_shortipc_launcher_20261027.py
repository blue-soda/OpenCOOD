import pathlib,json,subprocess,os,time,hashlib,tempfile
from multiprocessing.connection import Listener
b=pathlib.Path('/data2/tzm/cp_snn_recovery_20261026');repo=b/'OpenCOOD';root=b/'capacity2304_fp32_20261027'
protocol=json.loads((root/'protocol.json').read_text());assert protocol['primary_budget']==2304
for name,digest in protocol['hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assignments={'c0_small':4,'c1_wide':6};readings={a:[] for a in assignments}
for arm in assignments:assert not (root/arm/'launch.json').exists()
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
files=['opencood/models/e3dsnn_snapshot_capacity.py','opencood/utils/e3dsnn_capacity.py','opencood/tools/run_e3dsnn_capacity.py','opencood/tools/train_e3dsnn_single.py']
hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}
for arm,gpu in assignments.items():
 deadline=time.time()+21600
 while len(readings[arm])<5:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()
  values=[x for x in raw if int(x.split(',')[0])==gpu][0];idx,uuid,free,util=[x.strip() for x in values.split(',')]
  if int(free)>=5000 and int(util)<=20:
   readings[arm].append(dict(unix=time.time(),gpu=gpu,uuid=uuid,free_mib=int(free),utilization=int(util)))
   time.sleep(2)
  else:
   readings[arm]=[]
   state='resource_timeout' if time.time()>=deadline else 'waiting_for_gpu'
   (root/'launch_queue.json').write_text(json.dumps(dict(state=state,arm=arm,gpu=gpu,reading=values,deadline=deadline,unix=time.time())))
   if state=='resource_timeout':raise RuntimeError('GPU resource deadline reached')
   time.sleep(60)
 f=root/arm;tmp=pathlib.Path('/data2/tzm/capipc27')/arm;tmp.mkdir(exist_ok=False)
 with tempfile.TemporaryDirectory(prefix='pymp-',dir=str(tmp)) as probe:
  address=str(pathlib.Path(probe)/'listener-12345678')
  assert len(address.encode())<104
  listener=Listener(address);listener.close()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENCOOD_CUDA_MEMORY_FRACTION='.14',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo),TMPDIR=str(tmp))
 command=['/data1/tzm/miniconda3/envs/opencood/bin/python','-u','-m','opencood.tools.run_e3dsnn_capacity','--mode','coordinate','--root',str(root),'--arm',arm]
 with (f/'launch.json').open('x') as marker:marker.write(json.dumps(dict(state='launching',command=command)))
 with (f/'pipeline.log').open('xb') as log:
  p=subprocess.Popen(command,cwd=str(repo),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 value=dict(pid=p.pid,gpu=gpu,unix=time.time(),commit=revision,source_hashes=hashes,command=command,admission=readings[arm],automatic_retry=False,tmpdir=str(tmp),socket_bind_preflight=True)
 (f/'launch.json').write_text(json.dumps(value,indent=2));print(arm,p.pid,flush=True)
(root/'launch_queue.json').write_text(json.dumps(dict(state='both_launched',unix=time.time())))
