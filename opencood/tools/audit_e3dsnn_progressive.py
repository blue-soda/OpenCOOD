"""Bounded real-frame interface audit; no progressive-model AP or training claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.launch_e3dsnn_hybrid_round import sha,read
from opencood.models.fuse_modules.e3dsnn_progressive import MessageContract,ProgressiveMembraneFusion,split_counts,count_readout
from opencood.utils.e3dsnn_transplant import encoder_digest
from opencood.utils.e3dsnn_response_distill import state_digest
from opencood.utils import count4_compact_packet as compact
from opencood.utils import quantized_feature_packet as legacy


def audit(root,base):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError('Fresh audit directory required')
    source=Path(base)/'e3dsnn_budget_round1_20261018'
    prior=read(source/'adapt10_summary.json');assert prior['passed']
    checkpoint=source/'budget_s_fusion_adapt10/best_ap70.pth'
    assert sha(checkpoint)==prior['groups']['s_fusion']['checkpoint_sha256']
    cfg=load_yaml(str(source/'s_fusion.yaml'))
    dataset=build_dataset(cfg,train=False);model=train_utils.create_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
    enc=encoder_digest(model);initial=state_digest(model);captured={}
    def hook(module,inputs):captured['precount']=inputs[0].detach().clone()
    handle=model.backbone_3d.conv_out[-1].register_forward_pre_hook(hook)
    report=dict(started_unix=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_checkpoint_sha256=sha(checkpoint),encoder_sha256=enc,frames=[],
        scope='Eight fixed validation frames, shared frozen Count4 encoder, untrained receiver, fixed spatial partitions. No AP claim. Algebra audit uses float64.')
    try:
        with torch.no_grad():
            for index in np.linspace(0,len(dataset)-1,8,dtype=int):
                data=train_utils.to_device(dataset.collate_batch_test([dataset[int(index)]]),torch.device('cuda'))['ego']
                ego=model.encode_lidar(data['processed_lidar'],1)
                assert torch.equal(count_readout(captured['precount']),ego.features)
                precount=ego.replace_feature(captured['precount']).dense().double()
                road=model.encode_lidar(data['road_lidar'],1)
                frame=data['frame_ids'][0];info=dataset.co_idx2info[frame]
                road_frame=Path(info['infrastructure_pointcloud_path']).stem
                contract=MessageContract(frame,road_frame,'infrastructure',enc,'shared-3d-output-v1',
                    tuple(int(n) for n in road.spatial_shape),tuple(model.origin),tuple(model.step),
                    tuple(int(x) for x in data['timestamps_us'][0].cpu().tolist()))
                pose=data['road_to_ego'].double()
                meta=dict(asdict(contract),source_to_ego=pose[0].cpu().tolist())
                options=dict(bits=3,keep_fraction=.25)
                coords=road.indices[:,1:].cpu().numpy();values=road.features.cpu().numpy()
                old=legacy.encode(coords,values,meta,'count4',options)
                new=compact.encode(coords,values,meta,options)
                a,b,h=compact.decode(new,options);x,y,z=legacy.decode(old,'count4',options)
                assert np.array_equal(a,x) and np.array_equal(b,y) and h==z and len(new)<=len(old)
                dense=torch.zeros_like(precount)
                address=torch.as_tensor(a.astype('int64'),device='cuda')
                dense[0,:,address[:,0],address[:,1],address[:,2]]=torch.as_tensor(b.T.copy(),device='cuda',dtype=torch.double)
                fusion=ProgressiveMembraneFusion().cuda().double().eval()
                one=fusion.begin(precount,contract,pose,enc,contract.channel_basis)
                split=fusion.begin(precount,contract,pose,enc,contract.channel_basis)
                fusion.receive(one,dense,contract,pose,0)
                packet_lengths=[]
                for k,part in enumerate(split_counts(dense,3)):
                    active=(part[0]!=0).any(0);c=torch.nonzero(active).cpu().numpy()
                    f=part[0,:,active].T.cpu().numpy()
                    m=dict(meta,packet_index=k,packet_count=3)
                    packet=compact.encode(c,f,m,dict(bits=3,keep_fraction=1.))
                    c,f,received=compact.decode(packet,dict(bits=3,keep_fraction=1.))
                    received_contract=MessageContract(**{key:tuple(received[key]) if key in ('shape_zyx','origin_xyz','step_xyz','timestamps_us') else received[key] for key in asdict(contract)})
                    rebuilt=torch.zeros_like(part);ad=torch.as_tensor(c.astype('int64'),device='cuda')
                    rebuilt[0,:,ad[:,0],ad[:,1],ad[:,2]]=torch.as_tensor(f.T.copy(),device='cuda',dtype=torch.double)
                    assert torch.equal(part,rebuilt)
                    received_pose=pose.new_tensor(received['source_to_ego'])[None]
                    fusion.receive(split,rebuilt,received_contract,received_pose,received['packet_index'])
                    packet_lengths.append(len(packet))
                error=float(((one.output+one.residual)-(split.output+split.residual)).abs().max())
                assert error<1e-10
                differing=int((one.output!=split.output).sum())
                # If accumulation roundoff crosses a threshold, identify rather than hide it.
                if differing:
                    u=one.output+one.residual;distance=(u-(torch.floor(u)+.5)).abs()
                    assert float(distance[one.output!=split.output].max())<1e-10
                report['frames'].append(dict(index=int(index),contract=asdict(contract),
                    compact_bytes=len(new),legacy_bytes=len(old),three_packet_bytes=packet_lengths,
                    wire_exact=True,precount_matches_legacy=True,integrated_current_max_error=error,
                    count_differences_at_rounding_boundary=differing))
                del fusion,one,split,precount,dense,part,rebuilt,ego,road
        assert state_digest(model)==initial
        report['passed']=True;report['finished_unix']=time.time()
        report['frozen_source_unchanged']=True
        (root/'audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:handle.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--base',default='/data1/tzm/workspace/diagnostics')
    args=parser.parse_args();audit(args.root,args.base)
