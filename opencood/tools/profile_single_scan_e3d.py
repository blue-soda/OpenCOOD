"""Full-stream arithmetic of the two single-scan E-3DSNN transfers, counted on observed convolution edges."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

from opencood.tools.train_single_scan_e3d import (SingleScanDetector, DairEgoStreamDataset, V2VEgoStreamDataset,
    domain_rows, eval_utils, load_yaml, sha256, stream_collate, to_device, write_json)


def coordinate_fan(x, y, layer):
    """Independent convolution-edge enumeration for one batch, as an audit of the library's rulebook."""
    inp=x.indices.cpu().numpy()[:,1:].astype('int64')
    out=y.indices.cpu().numpy()[:,1:].astype('int64')
    dims=np.asarray(x.spatial_shape)
    flat=lambda a:(a[:,0]*dims[1]+a[:,1])*dims[2]+a[:,2]
    keys=flat(inp)
    order=keys.argsort()
    ordered=keys[order]
    hits=np.zeros(len(keys),dtype='int64')
    for offset in itertools.product(*(range(k) for k in layer.kernel_size)):
        coords=out*layer.stride-np.asarray(layer.padding)+np.asarray(offset)*layer.dilation
        valid=((coords>=0)&(coords<dims)).all(1)
        q=flat(coords[valid])
        at=np.minimum(ordered.searchsorted(q),len(order)-1)
        np.add.at(hits,order[at[ordered[at]==q]],1)
    return torch.as_tensor(hits,device=x.features.device)


def axis_fan(n, output, kernel, stride, pad, dilation, transpose, device):
    pos=torch.arange(n,device=device)
    fan=torch.zeros_like(pos)
    for offset in range(kernel):
        if transpose:
            index=pos*stride-pad+offset*dilation
            valid=(index>=0)&(index<output)
        else:
            index=pos+pad-offset*dilation
            valid=(index>=0)&(index<output*stride)&(index.remainder(stride)==0)
        fan+=valid
    return fan


class Arithmetic:
    columns=('edge_macs','nonzero_macs','analog_nonzero_macs','coded_nonzero_visits','unit_adds','binary_adds')
    def __init__(self,net,arm):
        self.arm=arm
        self.frame=0
        self.enabled=True
        self.totals={}
        self.sparse_cache={}
        self.dense_cache={}
        self.checked=set()
        for name,m in net.named_modules():
            if isinstance(m,(spconv.SubMConv3d,spconv.SparseConv3d,nn.Conv2d,nn.ConvTranspose2d)):
                self.totals[name]=torch.zeros(6,dtype=torch.int64,device='cuda')
                m.register_forward_hook(self.hook(name))
        assert self.totals

    def hook(self,name):
        coded=self.arm=='spike' and name not in ('encoder.conv_input.0','encoder.conv_out.0','cls','box','direction')
        def count(m,args,y):
            if not self.enabled:
                return
            x=args[0]
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                key=(x.shape[-2:],y.shape[-2:],m.kernel_size,m.stride,m.padding,m.dilation,isinstance(m,nn.ConvTranspose2d))
                if key not in self.dense_cache:
                    axes=[axis_fan(x.shape[2+j],y.shape[2+j],m.kernel_size[j],m.stride[j],m.padding[j],m.dilation[j],key[-1],x.device) for j in (0,1)]
                    fan=axes[0][:,None]*axes[1][None,:]
                    with torch.enable_grad():
                        probe=torch.ones((1,1,*x.shape[-2:]),device=x.device,requires_grad=True)
                        weight=probe.new_ones((1,1,*m.kernel_size))
                        if key[-1]:
                            out=nn.functional.conv_transpose2d(probe,weight,stride=m.stride,padding=m.padding,
                                                              output_padding=m.output_padding,dilation=m.dilation)
                        else:
                            out=nn.functional.conv2d(probe,weight,stride=m.stride,padding=m.padding,dilation=m.dilation)
                        assert torch.equal(torch.autograd.grad(out.sum(),probe)[0][0,0].long(),fan),name
                    self.dense_cache[key]=fan
                fan=self.dense_cache[key]
                v=x[0]
                channel=0
                cin,cout=m.in_channels,m.out_channels
            else:
                if m.indice_key not in self.sparse_cache:
                    book=y.indice_dict[m.indice_key]
                    if hasattr(book,'pair_fwd'):
                        pairs=book.pair_fwd
                        ix=pairs[pairs>=0].long()
                        fan=torch.bincount(ix,minlength=len(x.features))
                        if self.frame==0:
                            assert torch.equal(fan,coordinate_fan(x,y,m)),name
                            self.checked.add(m.indice_key)
                    else:
                        fan=coordinate_fan(x,y,m)
                    self.sparse_cache[m.indice_key]=fan
                fan=self.sparse_cache[m.indice_key]
                v=x.features
                channel=1
                cin,cout=m.in_channels,m.out_channels
            assert m.groups==1
            total=self.totals[name]
            total[0]+=fan.sum()*cin*cout
            active=((v!=0).sum(channel)*fan).sum()*cout
            total[1]+=active
            if coded:
                if self.frame<2:
                    assert ((v==v.round())&(v>=0)&(v<=4)).all(),name
                total[3]+=active
                total[4]+=(v.sum(channel,dtype=torch.int64)*fan).sum()*cout
                # Codes 1,2,4 each use one prescaled weight; code 3 needs two.
                total[5]+=active+((v==3).sum(channel)*fan).sum()*cout
            else:
                total[2]+=active
        return count


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.fit/'contract.json').read_text())
    state=a.fit/'train_04.pt'
    expected=json.loads((a.fit/'val_04.json').read_text())
    assert sha256(state)==json.loads((a.fit/'train_04.json').read_text())['sha']
    cfg=load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['preprocess']['args']['voxel_size']=receipt['voxel_size']
    cfg['data_augment']=[]
    net=SingleScanDetector(np.array(receipt['grid']),a.arm).cuda().eval()
    net.load_state_dict(torch.load(state,map_location='cpu'),strict=True)
    meter=Arithmetic(net,a.arm)
    rows=domain_rows(a,'val')
    ds=(DairEgoStreamDataset if a.domain=='dair' else V2VEgoStreamDataset)(rows,cfg,303)
    loader=DataLoader(ds,batch_size=1,num_workers=2,collate_fn=stream_collate)
    def empty_stats():
        return {t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)}
    stats=empty_stats()
    plain_stats=empty_stats()
    raw_max=0.
    raw_changed=0
    prev=torch.zeros(6,dtype=torch.int64,device='cuda')
    with (a.out/'frames.jsonl').open('x',buffering=1) as trace:
        for i,sample in enumerate(loader):
            meter.frame=i
            meter.sparse_cache.clear()
            ego=to_device(sample['ego'],torch.device('cuda'))
            pred,_=net(ego)
            row=sample['meta']
            total=torch.stack(list(meter.totals.values())).sum(0)
            rec=dict(scene=row['scene'],frame=row['frame'],counts=(total-prev).tolist())
            prev=total
            if row['supervised']:
                meter.enabled=False
                plain,_=net(ego)
                meter.enabled=True
                discrepancy=max(float((pred[k]-plain[k]).abs().max()) for k in pred)
                raw_max=max(raw_max,discrepancy)
                raw_changed+=int(discrepancy>0)
                rec['hook_max_delta']=discrepancy
                gt=ds.post.generate_gt_bbx({'ego':ego})
                for tag,prediction,acc in [('',pred,stats),('plain_',plain,plain_stats)]:
                    boxes,scores=ds.post.post_process({'ego':ego},{'ego':prediction})
                    for t in (.5,.7):
                        per={t:dict(tp=[],fp=[],gt=0,score=[])}
                        eval_utils.caluclate_tp_fp(boxes,scores,gt,per,t)
                        for k in per[t]:
                            acc[t][k]+=per[t][k]
                        rec[tag+'ap%d'%round(t*100)]=per[t]
            trace.write(json.dumps(rec)+'\n')
    aps=[eval_utils.calculate_ap(stats,t)[0] for t in (.5,.7)]
    plain_aps=[eval_utils.calculate_ap(plain_stats,t)[0] for t in (.5,.7)]
    assert stats[.7]['gt']==plain_stats[.7]['gt']==expected['gt']
    write_json(a.out/'result.json',dict(parent_sha=sha256(state),scans=len(rows),gt=stats[.7]['gt'],ap50=aps[0],ap70=aps[1],
        plain_ap50=plain_aps[0],plain_ap70=plain_aps[1],checkpoint_eval=expected,
        raw_max_hook_on_off=raw_max,raw_changed_labeled_frames=raw_changed,
        columns=list(meter.columns),layers={k:v.tolist() for k,v in meter.totals.items()},
        rulebook_keys_verified=sorted(meter.checked),scope='Conv edges excluding padding and absent sparse sites. No voxelization, BN, activation, routing, memory, time or power. Unit/binary adds conditional on receiving hardware.'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','datasets','fit','out'):
        p.add_argument(key,type=Path)
    p.add_argument('domain',choices=('dair','v2v'))
    p.add_argument('arm',choices=('spike','continuous'))
    run(p.parse_args())
