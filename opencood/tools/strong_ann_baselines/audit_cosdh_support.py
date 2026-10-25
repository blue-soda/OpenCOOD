"""Two-frame diagnostic: latent support required by CoSDH's post-warp mask.

No final wire-byte/AP claim. Includes two 3x3 decoder convolutions and a one-cell
roundoff guard beyond bilinear contributors; proves retained feature equivalence.
"""
import os,json,sys,time,hashlib,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='4'
root=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');out=root/'cosdh_support_audit01'
if out.exists():raise RuntimeError('Output exists')
for _ in range(5):
    line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().splitlines()[4]
    index,free,util=[int(x.strip()) for x in line.split(',')]
    if free<8000 or util>30:raise RuntimeError('GPU4 admission failed')
    time.sleep(1)
out.mkdir();repo=Path('/data1/zhh/gzc/CoSDH');sys.path.insert(0,str(repo))
import torch
import torch.nn.functional as F
import numpy as np
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
torch.set_num_threads(1);torch.manual_seed(303);np.random.seed(303);torch.cuda.set_per_process_memory_fraction(.25)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
cfg=yaml_utils.load_yaml(str(repo/'opencood/logs/dairv2x_cosdh/config.yaml'))
data=Path('/data1/zhh/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure')
cfg.update(data_dir=str(data),root_dir=str(data/'train.json'),validate_dir=str(root/'common_val.json'),test_dir=str(root/'common_val.json'),comm_range=1000000)
cfg['fusion']['core_method']='intermediatelate'
model=train_utils.create_model(cfg);model.load_state_dict(torch.load(str(repo/'opencood/logs/dairv2x_cosdh/net_epoch11.pth'),map_location='cpu'),strict=True);model.cuda().eval()
dataset=build_dataset(cfg,visualize=True,train=False);cache={};handles=[]
def latent_hook(stage):
    def hook(module,inputs):cache[(stage,'latent')]=inputs[0].detach().clone()
    return hook
def fuse_hook(stage):
    def hook(module,inputs):
        cache[(stage,'decoded')]=inputs[0].detach().clone();cache[(stage,'matrix')]=inputs[3].detach().clone()
    return hook
def mask_hook(stage):
    def hook(module,inputs,output):cache[(stage,'mask')]=output[0].detach().clone()
    return hook
for stage in range(3):
    handles += [model.naive_compressor_list[stage].decoder.register_forward_pre_hook(latent_hook(stage)),
                model.fusion_net[stage].register_forward_pre_hook(fuse_hook(stage)),
                model.fusion_net[stage].naive_communication.register_forward_hook(mask_hook(stage))]
records=[]
try:
    with torch.no_grad():
        for index in range(2):
            batch=train_utils.to_device(dataset.collate_batch_test([dataset[index]]),torch.device('cuda'))
            assert batch['ego']['record_len'].tolist()==[2]
            cache.clear();model(batch['ego'])
            for stage in range(3):
                latent=cache[(stage,'latent')];decoded=cache[(stage,'decoded')];matrix=cache[(stage,'matrix')][0,0,1:2]
                mask=cache[(stage,'mask')][1:2];h,w=decoded.shape[-2:]
                if mask.shape[-2:]!=(h,w):mask=F.interpolate(mask,size=(h,w),mode='bilinear',align_corners=False)
                active=mask[0,0]!=0
                grid=F.affine_grid(matrix,(1,decoded.shape[1],h,w),align_corners=False)[0]
                gx=((grid[...,0]+1)*w-1)/2;gy=((grid[...,1]+1)*h-1)/2
                x0=torch.floor(gx[active]).long();y0=torch.floor(gy[active]).long()
                source=torch.zeros((h,w),device=latent.device)
                for dx in (0,1):
                    for dy in (0,1):
                        x=x0+dx;y=y0+dy;ok=(x>=0)&(x<w)&(y>=0)&(y<h);source[y[ok],x[ok]]=1
                support=F.max_pool2d(source[None,None],7,stride=1,padding=3)[0,0].bool()
                sparse=latent.clone();sparse[1]*=support
                assert torch.equal(latent,latent.half().float())
                candidate=model.naive_compressor_list[stage].decoder(sparse)
                expected=warp_affine_simple(decoded[1:2],matrix,(h,w))*mask
                restored=warp_affine_simple(candidate[1:2],matrix,(h,w))*mask
                error=float((expected-restored).abs().max());exact=torch.equal(expected,restored)
                record={'frame_id':dataset.split_info[index],'stage':stage,'latent_shape':list(latent.shape),'target_selected':int(active.sum()),'source_bilinear_sites':int(source.sum()),'source_latent_sites_with_decoder_halo':int(support.sum()),'fp16_value_bytes_only':int(support.sum())*latent.shape[1]*2,'max_abs_masked_feature_error':error,'masked_features_exact':exact}
                records.append(record)
                if error>1e-5:raise ValueError('Dependency closure verification failed: '+str(record))
    result={'passed':True,'frames':2,'rows':records,'not_wire_bytes':True,'not_final_detection_parity':True,'decoder_radius':2,'numerical_guard_radius':1,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as exc:
    (out/'failure.json').write_text(json.dumps({'error':repr(exc),'rows':records},indent=2));raise
finally:
    for h in handles:h.remove()
