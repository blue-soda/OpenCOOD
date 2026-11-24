"""Single-scan E-3DSNN 3D/BEV backbone transfer with the common local anchor task."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from easydict import EasyDict

from opencood.data_utils.int_ego_stream import DairEgoStreamDataset, stream_collate, sha256
from opencood.data_utils.int_v2v_ego_stream import V2VEgoStreamDataset
from opencood.models.sub_modules.e3dsnn.backbone_3d import Multispike, VoxelBackBone8x_3dv_snn
from opencood.models.sub_modules.e3dsnn.backbone_2d import BaseBEVBackbone_spike
from opencood.tools.compare_event_lidar_history import domain_rows
from opencood.tools.train_int_external_memory import PointPillarDirLoss, load_yaml, to_device, write_json
from opencood.utils import eval_utils


def continuous_activations(parent):
    for name, module in parent.named_children():
        if isinstance(module, Multispike):
            setattr(parent, name, nn.Hardtanh(0., 4.))
        else:
            continuous_activations(module)


class SingleScanDetector(nn.Module):
    def __init__(self, grid, arm):
        super().__init__()
        self.encoder = VoxelBackBone8x_3dv_snn({}, 4, grid)
        cfg = EasyDict(LAYER_NUMS=[5,5], LAYER_STRIDES=[1,2], NUM_FILTERS=[64,128],
                       UPSAMPLE_STRIDES=[1,2], NUM_UPSAMPLE_FILTERS=[128,128])
        self.bev = BaseBEVBackbone_spike(cfg, 256)
        self.cls = nn.Conv2d(256,2,1)
        self.box = nn.Conv2d(256,14,1)
        self.direction = nn.Conv2d(256,4,1)
        nn.init.constant_(self.cls.bias,-np.log(99.))
        nn.init.normal_(self.box.weight,std=.001)
        self.sites = sum(isinstance(m,Multispike) for m in self.modules())
        assert self.sites > 30
        if arm == 'continuous':
            continuous_activations(self)
            assert not any(isinstance(m,Multispike) for m in self.modules())

    def forward(self, ego):
        p = ego['processed_lidar']
        mean = p['voxel_features'].sum(1) / p['voxel_num_points'].clamp(min=1).reshape(-1,1)
        sparse = self.encoder(dict(batch_size=1, voxel_features=mean.contiguous(),
                                   voxel_coords=p['voxel_coords']))['encoded_spconv_tensor']
        dense = sparse.dense()
        feat = self.bev(dict(spatial_features=dense.flatten(1,2)))['spatial_features_2d']
        assert feat.shape[2:] == ego['anchor_box'].shape[:2]
        return dict(psm=self.cls(feat),rm=self.box(feat),dm=self.direction(feat)), sparse.features


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    cfg = load_yaml(str(a.root/'inputs/config.yaml'))
    # load_yaml already derived anchor coordinates from the original 0.4 m grid.
    # Only voxelization changes. Stride eight at 0.1 m gives the same 0.8 m output.
    size = [.1,.1,.125]
    cfg['preprocess']['args']['voxel_size'] = size
    cfg['data_augment'] = []
    bounds = np.asarray(cfg['preprocess']['cav_lidar_range'])
    grid = np.rint((bounds[3:] - bounds[:3]) / size).astype(int)
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    net = SingleScanDetector(grid,a.arm).cuda()
    initial=hashlib.sha256()
    for k,v in net.state_dict().items():
        initial.update(k.encode())
        initial.update(v.detach().cpu().numpy().tobytes())
    opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.0001)
    loss_fn=PointPillarDirLoss(cfg['loss']['args']).cuda()
    raw={s:[r for r in domain_rows(a,s) if r['supervised']] for s in ('train','val')}
    assert tuple(map(len,raw.values())) == ((4678,1738) if a.domain=='dair' else (6958,748))
    write_json(a.out/'contract.json',dict(domain=a.domain,arm=a.arm,voxel_size=size,grid=grid.tolist(),
        initial_sha=initial.hexdigest(),activation_sites=net.sites,parameters=sum(p.numel() for p in net.parameters()),
        head='single-stage local anchor box/class/direction; no ROI refinement',seed=303,epochs=a.epochs,
        initialization='from scratch for both arms',training='AdamW 0.0005 constant; train BN running statistics'))
    dataset=DairEgoStreamDataset if a.domain=='dair' else V2VEgoStreamDataset
    for epoch in range(1,a.epochs+1):
        for split in ('train','val'):
            train=split=='train'
            if not train and epoch not in (1,a.epochs):
                continue
            order=np.arange(len(raw[split]))
            if train:
                np.random.RandomState(302+epoch).shuffle(order)
            ds=dataset([raw[split][i] for i in order],cfg,303)
            loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=stream_collate)
            stem='%s_%02d' % (split,epoch)
            stats={t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)}
            losses=[]
            net.train(train)
            with (a.out/(stem+'.jsonl')).open('x',buffering=1) as out, torch.set_grad_enabled(train):
                for step,sample in enumerate(loader):
                    ego=to_device(sample['ego'],torch.device('cuda'))
                    if train:
                        opt.zero_grad()
                    pred,encoded=net(ego)
                    row=sample['meta']
                    rec=dict(scene=row['scene'],frame=row['frame'])
                    if train:
                        loss=loss_fn(pred,ego['label_dict'])
                        loss.backward()
                        grad=nn.utils.clip_grad_norm_(net.parameters(),10.)
                        assert torch.isfinite(loss) and torch.isfinite(grad)
                        opt.step()
                        losses.append(float(loss.detach()))
                        rec.update(loss=losses[-1],grad=float(grad))
                        if step % 100 == 0:
                            activation=encoded.detach()
                            if a.arm=='spike':
                                assert ((activation==activation.round()) & (activation>=0) & (activation<=4)).all()
                            rec.update(active=float((activation!=0).float().mean()),
                                       first_conv_grad=float(net.encoder.conv_input[0].weight.grad.norm()))
                    else:
                        gt=ds.post.generate_gt_bbx({'ego':ego})
                        boxes,scores=ds.post.post_process({'ego':ego},{'ego':pred})
                        for t,k in ((.5,'ap50'),(.7,'ap70')):
                            per={t:dict(tp=[],fp=[],score=[],gt=0)}
                            eval_utils.caluclate_tp_fp(boxes,scores,gt,per,t)
                            rec[k]=per[t]
                            for field in stats[t]:
                                stats[t][field]+=per[t][field]
                    out.write(json.dumps(rec)+'\n')
            if train:
                checkpoint=a.out/(stem+'.pt')
                torch.save(net.state_dict(),str(checkpoint))
                summary=dict(steps=len(losses),loss=float(np.mean(losses)),sha=sha256(checkpoint))
            else:
                summary=dict(scans=len(ds),gt=stats[.7]['gt'],ap50=eval_utils.calculate_ap(stats,.5)[0],
                             ap70=eval_utils.calculate_ap(stats,.7)[0])
            write_json(a.out/(stem+'.json'),summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','datasets','out'):
        parser.add_argument(key,type=Path)
    parser.add_argument('domain',choices=('dair','v2v'))
    parser.add_argument('arm',choices=('spike','continuous'))
    parser.add_argument('--epochs',type=int,default=4)
    run(parser.parse_args())
