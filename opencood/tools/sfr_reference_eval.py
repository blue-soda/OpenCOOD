"""Explicit baseline checkpoint on the exact SFR development reader and sampling seed."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,os.getcwd())
from opencood.tools.sfr_run import seed,digest,plain
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.tools.train_utils import create_model,to_device
from opencood.utils import eval_utils


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','split_file','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--p',type=float,default=.3);p.add_argument('--max_steps',type=int,default=0)
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=False)
    cfg=load_yaml(args.config);cfg['validate_dir']=args.split_file;cfg['binomial_p']=args.p
    cfg['ectra_ego_history']=dict(enabled=True,max_age_ms=100)
    seed(303);dataset=SfrDAIRDataset(copy.deepcopy(cfg),False,False)
    model=create_model(cfg);model.load_state_dict(torch.load(args.checkpoint,map_location='cpu'),strict=True);model.cuda().eval()
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4,collate_fn=dataset.collate_batch_test,generator=torch.Generator().manual_seed(303))
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),config=plain(cfg),checkpoint_sha256=digest(args.checkpoint),
        split_sha256=digest(args.split_file),source_sha256={str(f):digest(f) for f in [Path(__file__),Path('opencood/models/point_pillar_codyntrust_cobevflow_baseline.py')]},
        note='Historical baseline trained on original train; development is not held out from baseline pretraining'),indent=2))
    stats={v:dict(tp=[],fp=[],gt=0,score=[]) for v in (.3,.5,.7)};identities=[]
    for index,batch in enumerate(loader):
        if args.max_steps and index>=args.max_steps:break
        if batch is None:continue
        batch=to_device(batch,'cuda')
        with torch.no_grad():
            output=model(batch['ego'],dataset);boxes,scores=dataset.post_processor.post_process(batch,dict(ego=output));gt=dataset.post_processor.generate_gt_bbx(batch)
            for threshold in stats:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats,threshold)
        meta=plain(batch['ego']['sfr_metadata'][0]);identities.append(dict(frame_id=meta['vehicle_frame'],intervals=plain(batch['ego']['past_k_time_interval']),timestamps=[[m['observation_us'] for m in row] for row in meta['observations']]))
        if index%100==0:print(json.dumps(dict(index=index,samples=len(identities))),flush=True)
    ap=eval_utils.eval_final_results(stats,str(out),dataset='d')
    result=dict(samples=len(identities),sample_time_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),smoke=bool(args.max_steps),**dict(zip(('ap30','ap50','ap70'),ap)))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
