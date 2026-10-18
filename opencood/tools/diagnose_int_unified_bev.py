"""Fixed-prefix teacher/student/reset control; never used for speed claims."""
import json
import torch
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.run_int_ego import loader,to_device
from opencood.utils import eval_utils


def diagnose(student,teacher,cfg,rows,out,frames=256):
    student.eval();teacher.eval()
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    names=['teacher','student','student_reset']
    ap={name:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in [.3,.5,.7]} for name in names}
    losses={name:[] for name in names};counts={name:[] for name in names}
    teacher_state=student_state=None;records=[]
    dataset,stream=loader(rows[:frames],cfg,2,303)
    with torch.no_grad(),(out/'diagnostic_frames.jsonl').open('x') as log:
        for sample in stream:
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
            reference,teacher_state,_=teacher.step(ego,teacher_state,meta,reset=meta['reset_before'])
            prediction,student_state,info=student.step(ego,student_state,meta)
            reset_prediction,_,_=student.step(ego,None,meta)
            outputs=dict(teacher=reference,student=prediction,student_reset=reset_prediction)
            record=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],
                        student_state_rms=float(student_state['value'].square().mean().sqrt()))
            if meta['supervised']:
                data={'ego':ego};gt=dataset.post.generate_gt_bbx(data)
                record['detections']={};record['gt_loss']={}
                for name,pred in outputs.items():
                    loss=float(criterion(pred,ego['label_dict']));losses[name].append(loss)
                    boxes,scores=dataset.post.post_process(data,{'ego':pred})
                    n=0 if boxes is None else len(boxes);counts[name].append(n)
                    record['detections'][name]=n;record['gt_loss'][name]=loss
                    for threshold in ap[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            log.write(json.dumps(record)+'\n');records.append(record)
    return dict(status='COMPLETED',scans=len(records),labels=sum(r['supervised'] for r in records),
        metrics={name:{'ap%d'%int(t*100):eval_utils.calculate_ap(ap[name],t)[0] for t in ap[name]} for name in names},
        mean_gt_loss={name:sum(values)/len(values) for name,values in losses.items()},
        mean_detections={name:sum(values)/len(values) for name,values in counts.items()},
        state_rms_range=[min(r['student_state_rms'] for r in records),max(r['student_state_rms'] for r in records)],
        caveats=['256-frame diagnostic prefix only; not full AP, convergence or latency evidence.',
                 'Reset is an inference intervention on a temporal checkpoint, not a separately trained single-frame baseline.',
                 'Teacher is an independent stream; no teacher values are injected into the student.'])
