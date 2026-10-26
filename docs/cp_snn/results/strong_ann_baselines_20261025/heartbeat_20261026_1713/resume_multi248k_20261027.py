import pathlib,subprocess,os,json,time,hashlib,shutil,traceback
import torch
b=pathlib.Path('/data2/tzm/cp_snn_recovery_20261026');root=b/'scheme2_joint_restart01';f=root/'multi248k'
r=b/'multi248k_resume_epoch2_01';r.mkdir(exist_ok=False)
def write(name,value):
 p=r/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 protocol=json.loads((root/'protocol.json').read_text())
 for name,digest in protocol['config_hashes'].items():assert sha(root/name)==digest
 assert sha(root/'source.pth')==protocol['source_sha256']
 assert json.loads((f/'status.json').read_text())['state']=='failed'
 state=torch.load(str(f/'train/last.pt'),map_location='cpu')
 assert state['epoch']==2 and state['scheduler']['last_epoch']==2
 assert state['selection_iou']=='0.7' and len(state['cuda_state'])==1
 assert sha(f/'train/last.pt')=='6ad70e589b8139acc32a13f5b6acbffc0cda252c3b97f3dde393069645d755f6'
 for p in [f/'status.json',f/'train/status.json',f/'train/progress.json',f/'train/metrics.jsonl',f/'train/last.pt',f/'train.log',f/'pipeline.log']:
  dest=r/'before_resume'/p.relative_to(f);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(p),str(dest))
 del state
 readings=[]
 for i in range(5):
  raw=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
  uuid,free,util=[x.strip() for x in raw.split(',')]
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits']).decode().splitlines()
  assert not [x for x in apps if x.split(',')[0].strip()==uuid] and int(free)>16000 and int(util)<=20
  readings.append(raw);time.sleep(2)
 tmp=r/'ipc_tmp';tmp.mkdir()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',TMPDIR=str(tmp),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCOOD_CUDA_MEMORY_FRACTION='.6',PYTHONPATH=str(b/'OpenCOOD'))
 initial=json.loads((f/'launch.json').read_text());deadline=initial['unix']+protocol['training_timeout_seconds']
 remaining=int(deadline-time.time());assert remaining>3600
 command=['/data1/tzm/miniconda3/envs/opencood/bin/python','-u','-m','opencood.tools.train_e3dsnn_single','--workers','2','--seed','20260929','--mode','train','--config',str(f/'train.yaml'),'--output',str(f/'train'),'--epochs','10','--selection-iou','0.7','--resume']
 info=dict(checkpoint_sha256=sha(r/'before_resume/train/last.pt'),resume_epoch=2,replay_epoch=3,discarded_partial_epoch=True,command=command,admission=readings,
  temporary_directory=str(tmp),original_training_deadline_unix=deadline,remaining_timeout_seconds=remaining,automatic_retry=False,
  code_commit=subprocess.check_output(['git','-C',str(b/'OpenCOOD'),'rev-parse','HEAD']).decode().strip(),
  preserved=['model','optimizer','scheduler','random_state','numpy_state','torch_state','cuda_state','loader_state','workers','pin_memory','batch_size','loss','selection_iou'])
 write('protocol.json',info)
 with (r/'train_resume.log').open('wb') as log:
  p=subprocess.Popen(['timeout','--kill-after=30',str(remaining)]+command,cwd=str(b/'OpenCOOD'),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  write('status.json',dict(state='training',pid=p.pid,unix=time.time()))
  (f/'status.json').write_text(json.dumps(dict(state='running',stage='train_resume_epoch2',supervisor=str(r),pid=p.pid),indent=2))
  code=p.wait()
 if code:raise RuntimeError('Training resume failed: '+str(code))
 assert json.loads((f/'train/status.json').read_text())['state']=='complete'
 write('status.json',dict(state='final_evaluation',unix=time.time()))
 with (r/'finish.log').open('wb') as log:
  subprocess.run([command[0],'-u','-m','opencood.tools.run_e3dsnn_snapshot_joint','--root',str(root),'--arm','multi248k','--mode','finish'],cwd=str(b/'OpenCOOD'),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
 write('status.json',dict(state='complete',unix=time.time()))
except Exception:
 write('status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False));raise
