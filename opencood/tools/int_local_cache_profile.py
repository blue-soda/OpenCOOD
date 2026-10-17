"""Train-only repair cost screen; no target features, AP or speed claims."""
import json
import time
from collections import defaultdict
import numpy as np
import torch
from opencood.models.sub_modules.int_early_cache import early_descriptor
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_local_cache import local_change_mask,local_requests
from opencood.models.sub_modules.int_cache_repair import stage_dependency_work,expand_tiles
from opencood.tools.run_int_ego import loader,to_device


def profile(a,model,cfg,rows,out,policy):
    assert cfg['model']['args']['base_bev_backbone']['layer_nums'] == [3,4,5]
    assert policy['gate']=='stage1' and policy['cache_kind']=='stages'
    threshold=policy['selected']['threshold']
    bounds=cfg['model']['args']['lidar_range']
    _,stream=loader(rows[:a.train_frames],cfg,4,303)
    state=cache=None;records=[];memo={};started=time.monotonic()
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,row in zip(stream,rows[:a.train_frames]):
            meta=sample['meta'];assert meta['frame']==row['frame']
            ego=to_device(sample['ego'],torch.device('cuda'));p=ego['processed_lidar']
            if p['voxel_features'].shape[0]:
                batch={k:p[k] for k in ['voxel_features','voxel_coords','voxel_num_points']}
                batch['batch_size']=1
                spatial=model.scatter(model.pillar_vfe(batch))['spatial_features']
            else:
                nx,ny,_=model.scatter.model_cfg['grid_size']
                spatial=model.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
            temporal,state,info=model.feature_memory.step(spatial,state,meta)
            first=model.backbone.resnet.layer0(temporal)
            desc=early_descriptor(first,4)
            h,w=first.shape[-2:];shapes=[((h+1)//2,(w+1)//2),((h+3)//4,(w+3)//4)]
            full=cache is None or bool(info['reset']) or cache['used'] or (int(meta['timestamp_us'])-cache['stamp'])/1e6>=.25
            entry=dict(frame=meta['frame'],segment=meta['segment'],full_refresh=full,cases={})
            if full:
                cache=dict(desc=desc,pose=meta['pose'],stamp=int(meta['timestamp_us']),used=False)
            else:
                aligned,valid=warp_bev(cache['desc'],cache['pose'],meta['pose'],bounds)
                change=local_change_mask(desc,aligned,valid,threshold)
                change=expand_tiles(change,4,first.shape[-2:])
                coverage=[]
                for shape in shapes:
                    _,covered=warp_bev(desc.new_ones((1,1,*shape)),cache['pose'],meta['pose'],bounds)
                    coverage.append(covered)
                entry['boundary_fractions']=[float((~v).float().mean()) for v in coverage]
                entry['stage1_changed_fraction']=float(change.float().mean())
                for scenario in ['boundary_only_lower_bound','content_and_boundary']:
                    changed=torch.zeros_like(change) if scenario.startswith('boundary') else change
                    for tile in [4,8,16]:
                        masks=local_requests(changed,*coverage,tile)
                        stats=[]
                        for i,mask in enumerate(masks):
                            key=(i,mask.cpu().numpy().tobytes())
                            if key not in memo:
                                work,_=stage_dependency_work(getattr(model.backbone.resnet,'layer%d'%(i+1)),
                                                             mask,first.shape[-2:] if i==0 else shapes[0])
                                memo[key]=dict(required_output_fraction=float(mask.float().mean()),**work)
                            stats.append(memo[key])
                        entry['cases'][scenario+'_tile%d'%tile]=stats
                cache['used']=True
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if len(records)%200==0:print('LOCAL TRAIN PROFILE',len(records),flush=True)
    assert len(records)==min(a.train_frames,len(rows))
    partial=[r for r in records if not r['full_refresh']]
    assert partial
    results={}
    for name in partial[0]['cases']:
        stages=[]
        for i in range(2):
            stages.append(dict(mean_partial_output_fraction=float(np.mean([r['cases'][name][i]['required_output_fraction'] for r in partial])),
                mean_partial_ideal_conv_fraction=float(np.mean([r['cases'][name][i]['ideal_conv_mac_fraction'] for r in partial]))))
        # Full frames cost all three stages. Partial frames still compute stage1.
        saving=len(partial)/len(records)*sum((1-s['mean_partial_ideal_conv_fraction'])*mac for s,mac in zip(stages,[28.0756224,35.5074048]))/84.2268672
        results[name]=dict(stages=stages,ideal_backbone_conv_saved_fraction=saving)
    return dict(status='COMPLETED',split='train',scans=len(records),partial_opportunities=len(partial),
        threshold=threshold,source_policy=policy,results=results,
        boundary_fraction_mean=[float(np.mean([r['boundary_fractions'][i] for r in partial])) for i in range(2)],
        stage1_changed_fraction_mean=float(np.mean([r['stage1_changed_fraction'] for r in partial])),
        seconds=time.monotonic()-started,
        caveats=['Training-only geometry/content diagnostic; no AP or actual sparse execution.',
                 'Current FM and stage1 execute; no current stage2/3 target or teacher is computed.',
                 'Boundary-only is a necessary-work diagnostic ignoring content changes, not a deployable policy.',
                 'Ideal deduplicated conv work excludes decoder/FM/warp/selector/launch/memory; no latency claim.',
                 'Tile4/8/16 are training cost candidates; thresholds stay frozen from the early-gate suite.'])
