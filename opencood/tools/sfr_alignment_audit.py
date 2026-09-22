"""Raw retained lidar alignment plots, independent of learned SFR representations."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,os.getcwd())
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.geometry import reader_to_reference,transform_points
from opencood.tools.sfr_run import seed,plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--split_file',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=False)
    cfg=load_yaml(args.config);cfg['validate_dir']=args.split_file;cfg['ectra_ego_history']=dict(enabled=True,max_age_ms=100)
    seed(303);dataset=SfrDAIRDataset(cfg,False,False);records=[]
    for index in (0,190,400):
        item=dataset[index]
        if item is None:continue
        data=dataset.collate_batch_test([item])['ego'];raw=data['processed_lidar'];k=data['past_k_time_interval'].numel()//2
        points=[]
        for agent in (0,1):
            mask=raw['voxel_coords'][:,0]==agent*k;v=raw['voxel_features'][mask,:,:3];n=raw['voxel_num_points'][mask]
            xyz=v[torch.arange(v.shape[1])[None]<n[:,None]]
            transform=reader_to_reference(data['pairwise_t_matrix'][0,agent,0]).to(xyz)
            xyz=transform_points(xyz,transform);points.append(xyz.numpy())
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        for ax in axes:
            for cloud,color,label in zip(points,('royalblue','darkorange'),('ego current','infra latest aligned')):
                cloud=cloud[::max(1,len(cloud)//20000)]
                ax.scatter(cloud[:,0],cloud[:,1],s=.4,alpha=.3,c=color,label=label)
            ax.set_aspect('equal');ax.set_xlabel('ego x / m');ax.set_ylabel('ego y / m');ax.legend(markerscale=6)
        axes[0].set_xlim(-100.8,100.8);axes[0].set_ylim(-40,40)
        axes[1].set_xlim(-30,50);axes[1].set_ylim(-25,25)
        meta=plain(data['sfr_metadata'][0]);fig.suptitle('Raw lidar alignment: ego '+meta['vehicle_frame']+' / infra '+meta['observations'][1][0]['frame_id'])
        fig.tight_layout();fig.savefig(out/('alignment_%d.png'%index),dpi=150);plt.close(fig)
        records.append(dict(index=index,metadata=meta,transform=plain(reader_to_reference(data['pairwise_t_matrix'][0,1,0])),point_counts=[len(p) for p in points]))
    (out/'records.json').write_text(json.dumps(records,indent=2));print(json.dumps(dict(samples=len(records))))


if __name__=='__main__':main()
