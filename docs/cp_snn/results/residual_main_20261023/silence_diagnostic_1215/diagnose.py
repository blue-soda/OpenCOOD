import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/data1/zhh/gzc/OpenCOOD')
import torch
from opencood.tools.run_e3dsnn_residual_main import load
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.fuse_modules.e3dsnn_residual_stream import select_symbols
r=Path('/data1/zhh/gzc/diagnostics/e3dsnn_residual_main_20261023')
out=r/'silence_diagnostic_20261023_1215'
folder,cfg,model=load(r,'residual_snn')
initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
current=torch.load(str(out/'snn_last.pt'),map_location='cpu')
data=build_dataset(cfg,train=False)
report={'checkpoint_epoch':current['epoch'],'checkpoint_sha256':hashlib.sha256((out/'snn_last.pt').read_bytes()).hexdigest(),'frames':[]}
for label,weights in [('initial',initial),('current',current['model'])]:
 model.load_state_dict(weights,strict=True);model.eval();model.packet_roundtrip=False
 for index in [0,1]:
  batch=train_utils.to_device(data.collate_batch_test([data[index]]),torch.device('cuda'));d=batch['ego']
  with torch.no_grad():
   sparse,src=model.encode_membrane(d['road_lidar'],1)
   drive=model.sender.drive(src)
   contract,meta=model.metadata(d,0,src.shape[2:])
   chunks,packet,events,proxy=model.generate(src,meta,4096)
   row=dict(label=label,index=index,source_max=float(src.abs().max()),source_nonzero=int(torch.count_nonzero(src)),drive_max=float(drive.abs().max()),drive_mean=float(drive.abs().mean()),potential_bound_16=float(drive.abs().max())*16,events=len(events),bytes=len(packet),steps=len(chunks),rate_proxy=float(proxy))
  report['frames'].append(row)
 # Detection-only gradient on first held-out frame: no optimizer step.
 model.set_stage('joint');model.eval();model.budget=4096
 criterion=train_utils.create_loss(cfg).cuda();model.zero_grad()
 output=model(d);loss=criterion(output,d['label_dict']);loss.backward()
 report[label+'_task_gradient']={prefix:sum(float(p.grad.abs().sum()) for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None) for prefix in ['sender.project.','sender.context.','sender.decode.','receiver.']}
report['weight_changes']={k:dict(max_abs=float(v.abs().max()),delta_max=float((v-initial[k]).abs().max())) for k,v in current['model'].items() if k.startswith('sender.')}
(out/'diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
