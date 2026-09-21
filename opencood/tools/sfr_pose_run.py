"""Independent SFR background cache, pose training and held-out evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from opencood.tools.sfr_run import seed, digest, plain
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.sfr_dair import SfrDAIRDataset
from opencood.models.sfr.sender import SfrSender
from opencood.models.sfr.calibration import BackgroundCalibration, background_points, pose_loss
from opencood.models.sfr.geometry import se2, transform_points, reader_to_reference
from opencood.tools.train_utils import to_device


def build_cache(args, cfg, output):
    sfr = yaml.safe_load(Path(cfg['base_sfr_config']).read_text())
    base = load_yaml(sfr['base_config'])
    base['root_dir' if args.training_data else 'validate_dir'] = args.split_file
    base['ectra_ego_history'] = dict(enabled=True, max_age_ms=sfr['history_max_age_ms'])
    seed(cfg['seed'] if args.training_data else cfg['eval_seed'])
    dataset = SfrDAIRDataset(copy.deepcopy(base), False, args.training_data)
    sender = SfrSender(base['model']['args'], sfr['sender']).cuda()
    sender.load_state_dict(torch.load(sfr['single_checkpoint'],map_location='cpu'),strict=True)
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.workers,collate_fn=dataset.collate_batch_test,
                        generator=torch.Generator().manual_seed(cfg['eval_seed']))
    manifest = dict(config=cfg,split_file=args.split_file,split_sha256=digest(args.split_file),
                    single_sha256=digest(sfr['single_checkpoint']),training_data=args.training_data,
                    source_sha256={str(p):digest(p) for p in Path('opencood/models/sfr').glob('*.py')},
                    data_config=plain(base),smoke=bool(args.max_steps),cache_version='sfr-background-2-forward',
                    note='Predicted foreground excluded. Retained geometry only, no GT/track labels. Corresponding causal ego history; calibration nominal.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows, skipped, missing_history = [], [], 0
    for index,batch in enumerate(loader):
        if args.max_steps and index>=args.max_steps: break
        if batch is None:
            skipped.append(index);continue
        batch=to_device(batch,'cuda');data=batch['ego'];k=data['past_k_time_interval'].numel()//2
        history=data['ectra_ego_history']
        with torch.no_grad():
            _, messages=sender.extract(data['processed_lidar'],2*k,data['anchor_box'],dataset.post_processor)
            _, references=sender.extract(history['processed_lidar'],k,data['anchor_box'],dataset.post_processor)
            used=set();count=0
            for frame in range(k):
                meta=plain(data['sfr_metadata'][0]['observations'][1][frame])
                if meta['frame_id'] in used:continue
                used.add(meta['frame_id'])
                if not bool(history['valid'][0,frame]):missing_history+=1;continue
                message,_=sender.wire_message(messages[k+frame],meta)
                source=background_points(message,reader_to_reference(data['pairwise_t_matrix'][0,1,frame]),sender.bounds,cfg['max_points'],cfg['foreground_margin_cells'])
                target=background_points(references[frame],history['to_current_ego'][0,frame],sender.bounds,cfg['max_points'],cfg['foreground_margin_cells'])
                rows.append(dict(source={key:value.cpu() for key,value in source.items()},target={key:value.cpu() for key,value in target.items()},
                                 meta=meta,history_age_ms=float(history['age_ms'][0,frame]),index=index))
                count+=1
                if count>=cfg['max_pairs_per_sample']:break
        if index%100==0:print(json.dumps(dict(index=index,pairs=len(rows),missing_history=missing_history)),flush=True)
    if not rows:raise RuntimeError('No real history pairs cached')
    torch.save(dict(pairs=rows,manifest=manifest),output/'background_cache.pth')
    summary=dict(pairs=len(rows),skipped_indices=skipped,missing_history=missing_history,
                 source_points=sum(len(x['source']['xyz']) for x in rows)/len(rows),target_points=sum(len(x['target']['xyz']) for x in rows)/len(rows))
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


def evaluate(model,rows,cfg,seed_value):
    generator=torch.Generator().manual_seed(seed_value)
    sums=dict(samples=0,usable=0,accepted=0,before_translation_m=0.,raw_translation_m=0.,applied_translation_m=0.,
              before_yaw_rad=0.,raw_yaw_rad=0.,applied_yaw_rad=0.,loss_sum=0.)
    reasons={}
    with torch.no_grad():
        for row in rows:
            delta=(torch.rand(3,generator=generator)*2-1)*torch.tensor([cfg['perturb_translation_m']]*2+[cfg['perturb_yaw_rad']])
            perturb=se2(delta.cuda());expected=torch.inverse(perturb)
            source=to_device(row['source'],'cuda');target=to_device(row['target'],'cuda')
            source=dict(source,xyz=transform_points(source['xyz'],perturb))
            applied,raw,stats=model(source,target)
            sums['samples']+=1;sums['usable']+=int(stats['usable']);sums['accepted']+=int(stats['accepted'])
            reasons[stats['reason']]=reasons.get(stats['reason'],0)+1
            # Recovery relative to the NOMINAL cross-view pair, not unknown true calibration GT.
            expected_yaw=torch.atan2(expected[1,0],expected[0,0])
            applied_yaw=torch.atan2(applied[1,0],applied[0,0])
            sums['before_translation_m']+=float(expected[:2,3].norm())
            sums['raw_translation_m']+=float((raw[:2]-expected[:2,3]).norm())
            sums['applied_translation_m']+=float((applied[:2,3]-expected[:2,3]).norm())
            sums['before_yaw_rad']+=float(expected_yaw.abs())
            sums['raw_yaw_rad']+=float(torch.atan2(torch.sin(raw[2]-expected_yaw),torch.cos(raw[2]-expected_yaw)).abs())
            sums['applied_yaw_rad']+=float(torch.atan2(torch.sin(applied_yaw-expected_yaw),torch.cos(applied_yaw-expected_yaw)).abs())
            if stats['usable']:sums['loss_sum']+=float(pose_loss(raw,expected,cfg['calibration']))
    result={key:value/max(sums['samples'],1) if key.endswith('_m') or key.endswith('_rad') else value for key,value in sums.items()}
    result.update(loss_per_usable=sums['loss_sum']/max(sums['usable'],1),reasons=reasons,
                  interpretation='Artificial perturbation recovery relative to nominal real cross-view pairs; not absolute calibration ground truth')
    return result


def learn_or_eval(args,cfg,output):
    data=torch.load(args.cache,map_location='cpu');rows=data['pairs']
    if data['manifest']['config']!=cfg:raise ValueError('Cache configuration mismatch')
    dev=torch.load(args.dev_cache,map_location='cpu') if args.dev_cache else data
    for cache in (data, dev):
        if cache['manifest'].get('cache_version') != 'sfr-background-2-forward':
            raise ValueError('Background cache predates the verified forward-coordinate contract')
    if args.mode=='train':
        if not args.dev_cache or args.dev_cache==args.cache:raise ValueError('Training needs an independent development cache')
        if dev['manifest']['split_sha256']==data['manifest']['split_sha256']:raise ValueError('Training/development split identity')
        fit_scenes={r['meta']['sequence_id'] for r in rows};dev_scenes={r['meta']['sequence_id'] for r in dev['pairs']}
        if fit_scenes & dev_scenes:raise ValueError('Background train/dev scene leakage')
    seed(cfg['seed']);model=BackgroundCalibration(cfg['calibration']).cuda()
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu')
        if saved['config']!=cfg:raise ValueError('Pose checkpoint configuration mismatch')
        model.load_state_dict(saved['model'],strict=True)
    if args.mode=='eval':
        if not args.checkpoint:raise ValueError('Evaluation requires checkpoint')
        result=evaluate(model.eval(),rows,cfg,cfg['eval_seed']);(output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.0001)
    manifest=dict(config=cfg,arguments=vars(args),train_cache_sha256=digest(args.cache),dev_cache_sha256=digest(args.dev_cache),
                  source_sha256=digest('opencood/models/sfr/calibration.py'),supervision='known_artificial_SE2_only',stage='B',started_at=time.time())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    initial=evaluate(model.eval(),dev['pairs'],cfg,cfg['eval_seed'])
    (output/'initial_dev.json').write_text(json.dumps(initial,indent=2));print(json.dumps(dict(initial_dev=initial)),flush=True)
    best=float('inf')
    for epoch in range(args.epochs or cfg['epochs']):
        model.train();count=0;total=0.;grad_max=0.
        for position,index in enumerate(torch.randperm(len(rows)).tolist()):
            if args.max_steps and position>=args.max_steps:break
            row=rows[index];source=to_device(row['source'],'cuda');target=to_device(row['target'],'cuda')
            delta=(torch.rand(3,device='cuda')*2-1)*source['xyz'].new_tensor([cfg['perturb_translation_m']]*2+[cfg['perturb_yaw_rad']])
            perturb=se2(delta);expected=torch.inverse(perturb)
            source=dict(source,xyz=transform_points(source['xyz'],perturb))
            _,raw,stats=model(source,target)
            if not stats['usable']:continue
            loss=pose_loss(raw,expected,cfg['calibration'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite calibration loss')
            optimizer.zero_grad();loss.backward()
            gradient=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            if not torch.isfinite(torch.tensor(gradient)):raise FloatingPointError('Nonfinite calibration gradient')
            grad_max=max(grad_max,gradient);torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            total+=float(loss.detach());count+=1
            if position%100==0:print(json.dumps(dict(epoch=epoch+1,position=position,loss=float(loss.detach()),gradient=gradient)),flush=True)
        if not count or not grad_max:raise RuntimeError('No effective pose supervision/gradient')
        result=evaluate(model.eval(),dev['pairs'],cfg,cfg['eval_seed'])
        summary=dict(epoch=epoch+1,train_valid_count=count,train_loss=total/count,gradient_max=grad_max,development=result,smoke=bool(args.max_steps))
        (output/('summary_epoch%d.json'%(epoch+1))).write_text(json.dumps(summary,indent=2))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,stage='pose',epoch=epoch+1,manifest=manifest)
        torch.save(state,output/('sfr_pose_epoch%d.pth'%(epoch+1)))
        if result['loss_per_usable']<best:
            best=result['loss_per_usable'];(output/'selected.json').write_text(json.dumps(dict(epoch=epoch+1,criterion='dev_raw_loss_per_usable',value=best)))
        print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cache','train','eval'],required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split_file');p.add_argument('--training_data',action='store_true');p.add_argument('--cache');p.add_argument('--dev_cache');p.add_argument('--checkpoint')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int);p.add_argument('--max_steps',type=int,default=0)
    args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    if args.mode=='cache':build_cache(args,cfg,out)
    else:learn_or_eval(args,cfg,out)
    (out/'completed.json').write_text(json.dumps(dict(finished_at=time.time())))


if __name__=='__main__':main()
