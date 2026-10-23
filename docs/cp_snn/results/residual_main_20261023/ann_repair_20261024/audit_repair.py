import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'/data1/zhh/gzc/OpenCOOD')
import torch
from opencood.tools.run_e3dsnn_residual_main import load,digest,check_frozen,frozen_state
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.fuse_modules.e3dsnn_residual_stream import ann_quant
r=Path('/data1/zhh/gzc/diagnostics/e3dsnn_residual_main_20261023');out=r/'ann_failure_20261024_0635'
result=subprocess.run([sys.executable,'-m','unittest','opencood.tools.test_e3dsnn_residual_stream','opencood.tools.test_e3dsnn_fusion'],cwd='/data1/zhh/gzc/OpenCOOD',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/'unit_tests.txt').write_bytes(result.stdout);assert result.returncode==0
folder,cfg,model=load(r,'residual_ann');frozen=frozen_state(model)
p=out/'residual_ann/train/last.pt';saved=torch.load(str(p),map_location='cpu')
assert saved['epoch']==7 and saved['protocol_sha256']==digest(r/'protocol.json')
model.load_state_dict(saved['model'],strict=True);check_frozen(model,frozen);model.eval();model.budget=4096
ds=build_dataset(cfg,train=False);report={'checkpoint_epoch':saved['epoch'],'checkpoint_sha256':digest(p),'frames':[]}
with torch.no_grad():
 for i in [0,1]:
  batch=train_utils.to_device(ds.collate_batch_test([ds[i]]),torch.device('cuda'));d=batch['ego']
  original=model.encode_membrane;cached={id(d[k]):original(d[k],1) for k in ['processed_lidar','road_lidar']}
  model.encode_membrane=lambda lidar,b:cached[id(lidar)]
  model.packet_roundtrip=False;a=model(d);records=model.last_records
  model.packet_roundtrip=True;b=model(d)
  assert all(torch.equal(a[k],b[k]) and torch.isfinite(a[k]).all() for k in ['psm','rm','dm'])
  assert 0<a['packet_bytes'][0]<=4096
  report['frames'].append({'index':i,'bytes':a['packet_bytes'][0],'events':records[0]['events'],'steps':records[0]['steps'],'direct_wire_exact':True})
  model.encode_membrane=original
v=torch.tensor([-2.**25,2.**25,-1e8,1e8],requires_grad=True);hard=v.clamp(-3,3).round()
old=v+(hard-v).detach();new=ann_quant(v);new.sum().backward()
assert torch.equal(new,hard) and torch.equal(v.grad,torch.ones_like(v))
report.update(synthetic_input=v.detach().tolist(),old_symbols=old.detach().tolist(),fixed_symbols=new.detach().tolist(),gradient=v.grad.tolist(),passed=True,optimizer_step=False,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/data1/zhh/gzc/OpenCOOD').decode().strip())
(out/'repair_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
