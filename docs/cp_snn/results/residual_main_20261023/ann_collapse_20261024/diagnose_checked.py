import os
os.environ.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'/data1/zhh/gzc/OpenCOOD')
import torch
from opencood.tools.run_e3dsnn_residual_main import load,digest,evaluate_budgets
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
def main():
 r=Path('/data1/zhh/gzc/diagnostics/e3dsnn_residual_main_20261023');out=r/'ann_collapse_20261024_1405'
 folder,cfg,model=load(r,'residual_ann');ds=build_dataset(cfg,train=False)
 saved=torch.load(str(out/'train/last.pt'),map_location='cpu');assert saved['epoch']==8
 best=torch.load(str(out/'train/best.pth'),map_location='cpu')
 report=dict(checkpoint_epoch=8,checkpoint_sha256=digest(out/'train/last.pt'),best_epoch=json.loads((out/'train/best_metrics.json').read_text())['epoch'],best_sha256=digest(out/'train/best.pth'),frames=[])
 report['parameters']={}
 for label,state in [('best',best),('epoch8',saved['model'])]:
  model.load_state_dict(state,strict=True);model.eval();model.packet_roundtrip=True
  report['parameters'][label]={prefix:{'max_abs':max(float(v.abs().max()) for k,v in state.items() if k.startswith(prefix) and v.is_floating_point()),'all_finite':all(bool(torch.isfinite(v).all()) for k,v in state.items() if k.startswith(prefix))} for prefix in ['backbone_3d.conv4.','backbone_3d.conv_out.','backbone_2d.','cls_head.','reg_head.','sender.','receiver.']}
  criterion=train_utils.create_loss(cfg).cuda()
  with torch.no_grad():
   for i in [0,1]:
    data=train_utils.to_device(ds.collate_batch_test([ds[i]]),torch.device('cuda'))
    for budget in [0,4096]:
     model.budget=budget;y=model(data['ego']);loss=criterion(y,data['ego']['label_dict'])
     report['frames'].append(dict(label=label,index=i,budget=budget,bytes=y['packet_bytes'],loss=float(loss),cls_absmax=float(y['psm'].abs().max()),reg_absmax=float(y['rm'].abs().max()),finite=all(bool(torch.isfinite(y[k]).all()) for k in ['psm','rm','dm'])))
 model.load_state_dict(saved['model'],strict=True);model.contract_identity=report['checkpoint_sha256']
 dest=out/'ego_validation_checked';dest.mkdir(exist_ok=False)
 report['epoch8_full_ego']=evaluate_budgets(model,ds,2,[0],dest)
 report['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/data1/zhh/gzc/OpenCOOD').decode().strip()
 (out/'diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
