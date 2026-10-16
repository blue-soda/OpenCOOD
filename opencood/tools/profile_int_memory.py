"""Sequential same-GPU stream profiling; includes diagnostics, not energy claims."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.tools.run_int_ego import loader, to_device


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=256)
    p.add_argument('--modes',nargs='+',default=['gru','spike_gru'])
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Preserve previous profile')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    report=dict(scope='Same-GPU sequential first validation scans; IO excluded; instrumentation included; no AP or energy estimate',frames=a.frames,models={})
    for mode in a.modes:
        d=root/('full_'+mode)
        args=json.loads((d/'effective_runtime.json').read_text())['arguments']
        best=json.loads((d/'best.json').read_text())
        config=load_yaml(args['config']);config['data_augment']=[]
        rows=json.loads(Path(args['manifest']).read_text())['val'][:a.frames]
        random.seed(303);np.random.seed(303);torch.manual_seed(303)
        model=PointPillarINT(config['model']['args'],mode).cuda().eval()
        model.load_state_dict(torch.load(best['checkpoint'],map_location='cpu')['model'],strict=True)
        if hasattr(model.feature_memory,'fusion'):model.feature_memory.fusion.profile_support=True
        events={};handles=[]
        def pre(name):
            def hook(module,inputs):
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                events[name]=pair;pair[0].record()
            return hook
        def post(name):
            def hook(module,inputs,output):events[name][1].record()
            return hook
        modules=dict(pillar=model.pillar_vfe,scatter=model.scatter,fusion=model.feature_memory.fusion,
                     backbone=model.backbone,classification=model.cls_head,regression=model.reg_head)
        if model.shrink_flag:modules['shrink']=model.shrink_conv
        if model.use_dir:modules['direction']=model.dir_head
        for name,module in modules.items():
            handles += [module.register_forward_pre_hook(pre(name)),module.register_forward_hook(post(name))]
        _,stream=loader(rows,config,0,303);state=None;records=[]
        with torch.no_grad():
            for index,sample in enumerate(stream):
                meta=sample['meta']
                if sample['ego'] is None:raise ValueError('Invalid profile frame')
                ego=to_device(sample['ego'],torch.device('cuda'));events.clear()
                torch.cuda.synchronize();tick=time.perf_counter()
                _,state,stats=model.step(ego,state,meta)
                torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                if index>=10:
                    records.append(dict(frame=meta['frame'],wall_ms=wall,
                        stages_ms={k:s.elapsed_time(e) for k,(s,e) in events.items()},stats=stats))
        summary={'wall_p50_ms':float(np.median([r['wall_ms'] for r in records])),
                 'stages_mean_ms':{k:float(np.mean([r['stages_ms'].get(k,0) for r in records])) for k in modules}}
        for key in ('emission_nonzero_fraction','active_cells','active_tiles8','readout_refresh_fraction','readout_refresh_tiles8'):
            if key in records[0]['stats']:summary[key]=float(np.mean([r['stats'][key] for r in records]))
        report['models'][mode]=dict(checkpoint=best,summary=summary,records=records)
        for h in handles:h.remove()
        del model,state,ego;torch.cuda.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in report['models'].items()}))


if __name__=='__main__':main()
