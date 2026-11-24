"""DAIR/V2V transfer of the Event-LiDAR author's geometric input eventifier.

The full current scan and up to two aligned historical sweeps enter one
PointPillars detector. Geometric and random arms retain the same number of
in-range history points for every source scan and target scan, before voxels.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from opencood.data_utils.int_ego_stream import DairEgoStreamDataset, stream_collate, sha256
from opencood.data_utils.int_v2v_ego_stream import V2VEgoStreamDataset
from opencood.tools.train_int_external_memory import (
    PointPillarINT, PointPillarDirLoss, freeze_bn_stats, full_rows, load_yaml, metadata,
    to_device, v2v_rows, write_json)
from opencood.utils import eval_utils, pcd_utils


def points(row):
    read = pcd_utils.pcd_to_np if row.get('dataset') == 'v2v_single' else lambda path: pcd_utils.read_pcd(path)[0]
    cloud = np.asarray(read(row['pcd']), dtype=np.float32)
    assert cloud.ndim == 2 and cloud.shape[1] == 4 and np.isfinite(cloud).all()
    return pcd_utils.mask_ego_points(cloud)


class LineEvents:
    height, width = 64, 2048

    def __init__(self, lib):
        self.lib = ctypes.CDLL(str(lib))
        ptr = ctypes.c_void_p
        self.lib.make_eventifier.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float]
        self.lib.make_eventifier.restype = ptr
        self.lib.release_eventifier.argtypes = [ptr]
        self.lib.release_eventifier.restype = None
        self.lib.next_events.argtypes = [ptr,ptr,ptr,ctypes.c_float,ptr]
        self.lib.next_events.restype = None
        self.model = None

    def reset(self):
        self.close()
        self.model = self.lib.make_eventifier(self.height,self.width,-30.,15.)

    def close(self):
        if self.model:
            self.lib.release_eventifier(self.model)
            self.model = None

    def apply(self,cloud,pose):
        xyz = cloud[:,:3]
        n = len(cloud)
        if not 0 < n <= self.height*self.width:
            raise ValueError(('scan exceeds fixed polar buffer',n))
        polar = np.zeros((self.width,self.height,3),np.float32)
        radius = np.linalg.norm(xyz,axis=1)
        valid = radius > 0
        lat = np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1]))
        polar.reshape(-1,3)[:n] = np.stack([np.arctan2(-xyz[:,1],xyz[:,0]),lat,radius],axis=1)
        polar = np.ascontiguousarray(polar.transpose(0,2,1))
        flags = np.zeros(self.width*self.height,np.int32)
        pose = np.ascontiguousarray(pose,np.float32)
        self.lib.next_events(self.model,pose.ctypes.data,polar.ctypes.data,.069,flags.ctypes.data)
        assert not flags[n:].any() and not flags[:n][~valid].any()
        return flags[:n].astype(bool), int(((lat < np.deg2rad(-30)) | (lat > np.deg2rad(15))).sum())


def domain_rows(a,split):
    if a.domain=='dair':
        return full_rows(a.root/'sequence_manifest.json',a.datasets,split)
    rows=v2v_rows(a.root/'v2v_official_deduplicated.jsonl',split)
    for row in rows:
        for key in ('pcd','annotation','label'):
            row[key]=str(a.datasets/Path(row[key]).relative_to('/data2/cyk/gzc/dataset'))
    return rows


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    model=LineEvents(a.lib)
    try:
        for split,limit in [('train',8),('val',24)]:
            rows=domain_rows(a,split)
            scenes={}
            for i,row in enumerate(rows):
                if row['supervised']:
                    scenes.setdefault(row['scene'],[]).append(i)
            targets=sorted(i for indices in scenes.values() for i in
                           [indices[j] for j in np.linspace(0,len(indices)-1,min(len(indices),limit),dtype=int)])
            histories={i:[j for j in range(max(0,i-2),i) if rows[j]['segment']==rows[i]['segment']] for i in targets}
            masks_needed=set(j for indices in histories.values() for j in indices)
            needed=set(masks_needed)|set(targets)
            needed.update(j-1 for j in masks_needed if j>0 and rows[j-1]['segment']==rows[j]['segment'])
            packed,meta={},{}
            last=None
            for i in sorted(needed):
                row=metadata(rows[i])
                reset=last!=i-1 or i==0 or rows[i-1]['segment']!=row['segment']
                if reset:
                    model.reset()
                cloud=points(row)
                mask,outside=model.apply(cloud,row['pose'])
                if reset:
                    assert mask.sum()==(np.linalg.norm(cloud[:,:3],axis=1)>0).sum()
                if i in masks_needed:
                    packed[str(i)]=np.packbits(mask)
                meta[i]=dict(frame=row['frame'],scene=row['scene'],segment=row['segment'],
                             pose=row['pose'],points=len(cloud),events=int(mask.sum()),outside_angle=outside,
                             ref_index=None if reset else i-1)
                last=i
            for j in masks_needed:
                if j>0 and rows[j-1]['segment']==rows[j]['segment']:
                    assert meta[j]['ref_index']==j-1
            np.savez(a.out/(split+'_masks.npz'),**packed)
            write_json(a.out/(split+'.json'),dict(targets=targets,histories=histories,meta=meta))
    finally:
        model.close()
    write_json(a.out/'ready.json',dict(source='scsrp/event-lidar 9b059e45815be12e5bf47930d4842039789ee3bf',
               binding=sha256(a.lib),algorithm='LineTemporalEventifier<float> + DiffVec(0.069 m)',
               grid=[64,2048,-30,15],split='up to 8/24 supervised train/validation targets per scene; deterministic equally spaced indices; no tests'))


class HistoryInput:
    def read_points(self,row):
        i=row['target_index']
        data=self.audit['meta']
        current=points(row)
        assert len(current)==data[str(i)]['points']
        collected=[current]
        counts=[]
        bounds=self.config['preprocess']['cav_lidar_range']
        for j in self.audit['histories'][str(i)]:
            src=self.raw[j]
            assert j<i and src['segment']==row['segment']
            past=points(src)
            meta=data[str(j)]
            assert len(past)==meta['points']
            transform=np.linalg.solve(data[str(i)]['pose'],meta['pose'])
            past[:,:3]=past[:,:3] @ transform[:3,:3].T + transform[:3,3]
            # Use the detector's crop and ego exclusion before matching budgets.
            ids=np.arange(len(past),dtype=np.float64)
            packed=np.column_stack([past,ids])
            valid=pcd_utils.mask_ego_points(pcd_utils.mask_points_by_range(packed,bounds))[:,-1].astype(int)
            fired=np.unpackbits(self.masks[str(j)])[:len(past)].astype(bool)[valid]
            selected=valid
            if self.arm=='geometric':
                selected=valid[fired]
            elif self.arm=='random':
                selected=np.sort(np.random.RandomState((303+i*1009+j) % 2**32).choice(valid,int(fired.sum()),replace=False))
            counts.append([j,len(valid),int(fired.sum()),len(selected)])
            collected.append(past[selected])
        row['history_counts']=counts
        row['current_count']=len(current)
        return np.concatenate(collected)


class DairHistory(HistoryInput,DairEgoStreamDataset):
    pass


class V2VHistory(HistoryInput,V2VEgoStreamDataset):
    pass


def trial(a):
    a.out.mkdir(parents=True,exist_ok=False)
    assert (a.prepared/'ready.json').is_file()
    cfg=load_yaml(str(a.root/'inputs/config.yaml'))
    cfg['data_augment']=[]
    torch.manual_seed(303)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    p1=a.root/'external_int556/p1.pth'
    assert sha256(p1)=='c5fdef2fcc2914a09bb527cee6723730e70850056e3115f3197122e69431efb9'
    net=PointPillarINT(cfg['model']['args'],'single').cuda()
    net.load_single_checkpoint(torch.load(p1,map_location='cpu'))
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    write_json(a.out/'contract.json',dict(arm=a.arm,domain=a.domain,p1=sha256(p1),
               prepared={split:sha256(a.prepared/(split+'.json')) for split in ('train','val')},seed=303))
    for split in ('train','val'):
        audit=json.loads((a.prepared/(split+'.json')).read_text())
        raw=domain_rows(a,split)
        order=list(audit['targets'])
        if split=='train':
            np.random.RandomState(303).shuffle(order)
        rows=[dict(raw[i],target_index=i) for i in order]
        ds=(DairHistory if a.domain=='dair' else V2VHistory)(rows,cfg,303)
        ds.audit,ds.raw,ds.arm=audit,raw,a.arm
        with np.load(a.prepared/(split+'_masks.npz')) as masks:
            ds.masks={k:masks[k] for k in masks.files}
        stream=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=stream_collate)
        train=split=='train'
        net.train(train)
        freeze_bn_stats(net)
        stats={t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)}
        losses=[]
        with (a.out/(split+'.jsonl')).open('x',buffering=1) as log,torch.set_grad_enabled(train):
            for sample in stream:
                row=sample['meta']
                ego=to_device(sample['ego'],torch.device('cuda'))
                pred,_,_=net.step(ego,None,row,reset=True)
                rec=dict(i=row['target_index'],scene=row['scene'],frame=row['frame'],current=row['current_count'],
                         history=row['history_counts'],voxels=len(ego['processed_lidar']['voxel_features']))
                if train:
                    opt.zero_grad()
                    loss=criterion(pred,ego['label_dict'])
                    loss.backward()
                    grad=nn.utils.clip_grad_norm_(net.parameters(),10.)
                    assert torch.isfinite(loss) and torch.isfinite(grad)
                    opt.step()
                    rec.update(loss=float(loss.detach()),grad=float(grad))
                    losses.append(rec['loss'])
                else:
                    gt=ds.post.generate_gt_bbx({'ego':ego})
                    boxes,scores=ds.post.post_process({'ego':ego},{'ego':pred})
                    per={t:dict(tp=[],fp=[],gt=0,score=[]) for t in stats}
                    for t in stats:
                        eval_utils.caluclate_tp_fp(boxes,scores,gt,per,t)
                        for k in stats[t]:
                            stats[t][k]+=per[t][k]
                    rec.update(ap50=per[.5],ap70=per[.7])
                log.write(json.dumps(rec)+'\n')
        if train:
            torch.save(net.state_dict(),str(a.out/'trained.pt'))
            write_json(a.out/'train.json',dict(updates=len(losses),loss=np.mean(losses),sha=sha256(a.out/'trained.pt')))
        else:
            write_json(a.out/'val.json',dict(scans=len(rows),gt=stats[.7]['gt'],
                       ap50=eval_utils.calculate_ap(stats,.5)[0],ap70=eval_utils.calculate_ap(stats,.7)[0]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','datasets','out'):
        p.add_argument(name,type=Path)
    p.add_argument('domain',choices=('dair','v2v'))
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare').add_argument('lib',type=Path)
    tr=sub.add_parser('trial')
    tr.add_argument('prepared',type=Path)
    tr.add_argument('arm',choices=('raw','geometric','random'))
    args=p.parse_args()
    (prepare if args.action=='prepare' else trial)(args)
