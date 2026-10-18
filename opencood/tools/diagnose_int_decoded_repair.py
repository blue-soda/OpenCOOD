"""Audit frozen teacher, decoded warp-only, and trained repair on identical frames."""
import json
from pathlib import Path
import torch
from opencood.models.sub_modules.int_light_repair import make_repairs
from opencood.models.sub_modules.int_decoded_repair import DecodedCacheDetector
from opencood.tools.train_int_light_repair import training_pair
from opencood.tools.run_int_ego import loader,to_device
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def diagnose(a,model,cfg,rows,out,provenance):
    saved=torch.load(a.repair_checkpoint,map_location='cpu')
    assert saved['contract']['repair_target']=='decoded'
    assert saved['contract']['teacher_checkpoint_sha256']==provenance['checkpoint']['sha256']
    repairs=make_repairs('decoded').cuda().eval();repairs.load_state_dict(saved['repairs'],strict=True)
    zero=make_repairs('decoded').cuda().eval()
    wrappers=[DecodedCacheDetector(model,cfg,0.,period=3,repair_modules=m) for m in [repairs,zero]]
    states=[None,None,None];captured={};train_cache=None;records=[]
    def first_hook(module,inputs,output):captured['first']=output.detach()
    def decoded_hook(module,inputs):captured['decoded']=inputs[0].detach()
    hooks=[model.backbone.resnet.layer0.register_forward_hook(first_hook),model.cls_head.register_forward_pre_hook(decoded_hook)]
    names=['teacher','trained','zero_residual']
    ap={name:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in [.3,.5,.7]} for name in names}
    dataset,stream=loader(rows[:a.frames],cfg,2,303)
    try:
        with torch.no_grad(),(out/'frames.jsonl').open('x') as log:
            for sample in stream:
                meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
                teacher,states[0],_=model.step(ego,states[0],meta)
                first=captured['first'];reference=captured['decoded']
                predictions=[teacher];infos=[]
                for i,wrapper in enumerate(wrappers,1):
                    pred,states[i],info=wrapper.step(ego,states[i],meta)
                    predictions.append(pred);infos.append(info)
                partial=infos[0]['local_partial'];parity=0.;feature_loss=None
                if partial:
                    values,loss=training_pair(first,[reference],teacher,train_cache,meta,repairs,
                        cfg['model']['args']['lidar_range'],0.,True,'decoded')
                    replay=dict(psm=model.cls_head(values[0]),rm=model.reg_head(values[0]))
                    parity=max(float((replay[k]-predictions[1][k]).abs().max()) for k in replay)
                    feature_loss=float(loss)
                else:
                    train_cache=dict(values=[reference],pose=meta['pose'],stamp=int(meta['timestamp_us']))
                record=dict(frame=meta['frame'],supervised=meta['supervised'],partial=partial,
                    train_inference_max_error=parity,feature_loss=feature_loss,
                    pred_mse={name:{k:float((pred[k]-teacher[k]).square().mean()) for k in ['psm','rm']}
                              for name,pred in zip(names[1:],predictions[1:])})
                if meta['supervised']:
                    data={'ego':ego};gt=dataset.post.generate_gt_bbx(data)
                    record['detections']={}
                    for name,pred in zip(names,predictions):
                        boxes,scores=dataset.post.post_process(data,{'ego':pred})
                        record['detections'][name]=0 if boxes is None else len(boxes)
                        for threshold in ap[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
                log.write(json.dumps(record)+'\n');records.append(record)
    finally:
        for hook in hooks:hook.remove()
    return dict(status='COMPLETED',checkpoint_sha256=digest(a.repair_checkpoint),scans=len(records),
        labels=sum(r['supervised'] for r in records),partial_frames=sum(r['partial'] for r in records),
        train_inference_max_error=max(r['train_inference_max_error'] for r in records),
        metrics={name:{'ap%d'%int(t*100):eval_utils.calculate_ap(ap[name],t)[0] for t in ap[name]} for name in names},
        caveats=['Diagnostic prefix with teacher and multiple passes; not timing or full-validation evidence.',
                 'Checkpoint is an immutable saved epoch; originating training may still be running.',
                 'Training replay uses independently captured teacher full anchors and the actual training_pair function.'])
