"""Paired dense/cache-aware adaptation from a fixed mature projected stage.

Two-epoch design diagnostic, not a convergence claim. Both arms have the same
parameters, initialization, optimizer, supervision and data. Only the student's
spatial recurrence during training differs. Validation always measures real
selected execution and its independent masked-dense control.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_compact_readout import compact_shrink
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_projected_spatial import ProjectedSpatialStage
from opencood.models.sub_modules.int_compact_projected_state import CompactProjectedSpatialStage
from opencood.models.sub_modules.int_basis_spatial_cache import SpatialBasisCacheDetector
from opencood.models.sub_modules.int_split_readout import SplitCompactReadout,SplitReadoutDetector
from opencood.models.sub_modules.int_light_repair import LightRepair,foreground_head_loss
from opencood.tools.evaluate_int_compact_repair import completed_best
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_cache_training import enable_cache_training,training_step
from opencood.tools.int_selection_audit import capture_step,divergence_audit
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def load_fixed(a):
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    base,cfg,val,source=setup(a);rb,readout=completed_best(a.readout_run)
    artifacts=[]
    for root,sha in ((a.memory_run,'1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'),
                     (a.repair_run,'9f0997e23752683aa3a8fa6f5b9c9e65ddd91c7a1f2dca89e1beb7500a755fbd'),
                     (a.projected_run,'2424236f88bafb04325ec07803ba5f99614c5ab3762e951eed365725f0b3af62')):
        root=Path(root);assert json.loads((root/'status.json').read_text())['status']=='COMPLETED'
        best=json.loads((root/'best.json').read_text());assert digest(best['checkpoint'])==best['sha256']==sha
        artifacts.append((best,torch.load(best['checkpoint'],map_location='cpu')))
    (mb,memory),(pb,repaired),(sb,spatial)=artifacts
    assert spatial['epoch']==sb['epoch']==0 and spatial['contract']['rank']==96
    assert memory['contract']['data']==repaired['contract']['data']==spatial['contract']['data']==source['contract']
    assert spatial['contract']['memory']==mb and spatial['contract']['repair']==pb
    assert spatial['contract']['readout']==repaired['contract']['readout']==rb
    assert memory['contract']['source']['sha256']==readout['contract']['source_sha256']==source['checkpoint']['sha256']
    base.shrink_conv=compact_shrink(base.shrink_conv,96)[0].cuda();base.shrink_conv.load_state_dict(readout['readout'],strict=True)
    base.feature_memory=projected_gru(base.feature_memory,32)[0].cuda();base.feature_memory.load_state_dict(memory['memory'],strict=True)
    stage=ProjectedSpatialStage(base.backbone.resnet.layer0,96).cuda().eval();stage.load_state_dict(spatial['stage'],strict=True)
    base.backbone.resnet.layer0=CompactProjectedSpatialStage(stage).cuda().eval();base.eval()
    split=SplitCompactReadout(base.shrink_conv,base.backbone.deblocks[0][0].out_channels).cuda().eval()
    repair=LightRepair(96,width=32).cuda().eval();repair.load_state_dict(repaired['repair'],strict=True)
    for model in (base,split,repair):
        for p in model.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(arguments['manifest']).read_text())['train']
    return base,split,repair,cfg,train,val,dict(source=source,memory=mb,readout=rb,repair=pb,spatial=sb)


def modes_for(base,student,split,repair,cfg):
    def region(model,execution='selected'):
        return SpatialBasisCacheDetector(model,cfg,split,repair,.75,confidence_threshold=.1,execution=execution)
    return dict(fixed=SplitReadoutDetector(base,cfg,split,3,repair),fixed_selected=region(base),
        full=SplitReadoutDetector(student,cfg,split,1),dense=SplitReadoutDetector(student,cfg,split,3,repair),
        selected=region(student),masked_dense=region(student,'masked_dense'))


def executed_macs(name,info,costs,stage):
    macs=costs[('full' if name=='full' else 'cached',bool(info['local_partial']))]
    if name in ('selected','fixed_selected') and not info['spatial_refresh']:
        cells=info['spatial_cells']
        mix=sum(cells*c.mix.weight.numel() for b in stage.blocks for c in (b.conv1,b.conv2))
        projection=sum(cells*b.conv2.project.weight.numel() for b in stage.blocks)
        macs=macs-mix-projection+info['spatial_extra_linear_macs']
    return macs


def engineering(base,student,split,repair,cfg,train):
    modes=modes_for(base,student,split,repair,cfg);states={k:None for k in modes};costs={};maximum=0.
    _,stream=loader(train[:32],cfg,0,303)
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));predictions={};infos={};counts={}
            for name,mode in modes.items():
                root=base if name.startswith('fixed') else student
                counter=ConvCounter(root,torch.nn.ModuleList([split,repair]))
                try:predictions[name],states[name],infos[name]=mode.step(ego,states[name],sample['meta'])
                finally:counter.close()
                counts[name]=counter.total
                if name in ('full','dense'):
                    key=('full' if name=='full' else 'cached',bool(infos[name]['local_partial']))
                    if key in costs:assert costs[key]==counter.total
                    costs[key]=counter.total
            for name,info in infos.items():
                expected=counts[name]+info.get('spatial_extra_linear_macs',0)
                assert expected==executed_macs(name,info,costs,student.backbone.resnet.layer0)
            for key in predictions['selected']:
                maximum=max(maximum,float((predictions['selected'][key]-predictions['masked_dense'][key]).abs().max()))
                assert torch.allclose(predictions['selected'][key],predictions['masked_dense'][key],atol=1e-3,rtol=1e-3)
    # Backpropagate across consecutive steps with detached own states. Probe
    # both refresh and partial paths without changing weights or BN buffers.
    mode=enable_cache_training(modes['masked_dense']);state=None;norms=[]
    _,stream=loader(train[:4],cfg,0,303)
    buffers={n:b.detach().clone() for n,b in student.named_buffers()}
    for sample in stream:
        ego=to_device(sample['ego'],torch.device('cuda'));student.zero_grad(set_to_none=True)
        pred,state,info=training_step(mode,ego,state,sample['meta'])
        loss=pred['rm'].square().mean()+pred['psm'].square().mean();loss.backward()
        values=[p for p in student.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in values)
        norms.append(float(torch.nn.utils.clip_grad_norm_(values,5.)))
        assert norms[-1]>0 and all(p.grad is None for p in student.parameters() if not p.requires_grad)
        assert not state['cache']['value'].requires_grad
    assert all(torch.equal(buffers[n],b) for n,b in student.named_buffers())
    student.zero_grad(set_to_none=True)
    return costs,dict(status='PASSED',frames=32,selected_masked_max_error=maximum,
        consecutive_gradient_norms=norms,only_factors_have_gradients=True,bn_unchanged=True,
        detached_own_spatial_fm_and_deep_states=True,actual_conv_function_macs_verified=True)


def evaluate(base,student,split,repair,cfg,rows,directory,costs,epoch):
    modes=modes_for(base,student,split,repair,cfg);states={k:None for k in modes}
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    metrics={k:dict(times=[],areas=[],projections=[],bytes=[],peaks=[],macs=0,refreshes=0) for k in modes};audits=[]
    dataset,stream=loader(rows,cfg,4,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));previous=dict(states);masks={}
            names=list(modes);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                if name in ('selected','masked_dense'):
                    pred,states[name],info,masks[name]=capture_step(modes[name],ego,states[name],sample['meta'])
                else:pred,states[name],info=modes[name].step(ego,states[name],sample['meta'])
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick);item=metrics[name]
                if index>=10:
                    item['times'].append(elapsed);item['peaks'].append(max(0,torch.cuda.max_memory_allocated()-before))
                assert all(torch.isfinite(v).all() for v in pred.values())
                item['macs']+=executed_macs(name,info,costs,student.backbone.resnet.layer0)
                item['areas'].append(info.get('spatial_mixer_executed_fraction',1.))
                item['projections'].append(info.get('spatial_projection_executed_fraction',1.))
                item['bytes'].append(info['state_bytes']+info['anchor_bytes']+info.get('spatial_state_bytes',0))
                item['refreshes']+=int(info.get('spatial_refresh',True))
                if sample['meta']['supervised']:
                    data=dict(ego=ego);boxes,scores=dataset.post.post_process(data,dict(ego=pred));gt=dataset.post.generate_gt_bbx(data)
                    for t in stats[name]:eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            if not torch.equal(masks['selected'],masks['masked_dense']):
                audit=divergence_audit(modes['selected'],modes['masked_dense'],ego,previous,sample['meta'],masks)
                audit.update(scans=index+1,segment=sample['meta']['segment'])
                audits.append(audit);write_json(directory/'recurrence_audits.json',dict(audits=audits))
                assert audit['same_input_masks_equal'] and audit['same_input_numerically_close'] and len(audits)<=128
            if (index+1)%200==0:write_json(directory.parent/'status.json',dict(status='RUNNING',phase='validation',epoch=epoch,scans=index+1))
    assert index+1==len(rows)
    results={k:dict(ap70=eval_utils.calculate_ap(stats[k],.7)[0],ap50=eval_utils.calculate_ap(stats[k],.5)[0],
        mean_ms=float(np.mean(v['times'])),p95_ms=float(np.percentile(v['times'],95)),
        conv_linear_gflops=2*v['macs']/len(rows)/1e9,mean_executed_mix_area=float(np.mean(v['areas'])),
        mean_executed_projection_area=float(np.mean(v['projections'])),mean_state_bytes=float(np.mean(v['bytes'])),
        max_incremental_cuda_peak_bytes=max(v['peaks']),spatial_refresh_frames=v['refreshes']) for k,v in metrics.items()}
    assert abs(results['fixed']['ap70']-.7559544809175783)<1e-8
    assert abs(results['fixed_selected']['ap70']-.7546151539838869)<1e-7
    if epoch==0:
        assert abs(results['full']['ap70']-.7567910703)<1e-7
        assert abs(results['dense']['ap70']-results['fixed']['ap70'])<1e-8
        assert abs(results['selected']['ap70']-results['fixed_selected']['ap70'])<1e-8
    report=dict(status='COMPLETED',scans=len(rows),labels=sum(r['supervised'] for r in rows),results=results,
        own_mask_disagreement_frames=len(audits),same_input_replay_passed_all=True,
        scope='Own independent recurrent modes. Fixed r96 e0 reference is NOT original INT. Selected is true skipped execution; masked_dense executes full convolutions with identical cache semantics. Period3/.25s, threshold.75, own confidence.1, FP32. Actual conv+functional-linear MACsx2 excludes nonconv operations. Same-GPU rotating model time excludes IO/H2D/postprocess; mask capture included, comparison/replay excluded. Peak is incremental above live six-mode allocations, not standalone VRAM.')
    write_json(directory/'report.json',report);return report


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    base,split,repair,cfg,train,val,provenance=load_fixed(a);torch.manual_seed(303)
    student=copy.deepcopy(base);stage=student.backbone.resnet.layer0;parameters=[]
    for block in stage.blocks:
        for layer in (block.conv1,block.conv2):
            for p in layer.parameters():p.requires_grad_(True);parameters.append(p)
    contract=dict(arm=a.arm,provenance=provenance,seed=303,epochs_max=2,
        trainable_parameters=sum(p.numel() for p in parameters),
        trainable='Only four r96 projection/spatial kernel pairs. Frozen stem, BN affine/statistics, FM, all downstream/readout/repair/heads. No new inference parameters/operators.',
        train='Same fixed initialization/AdamW1e-5 wd1e-4 clip5/ordered train8593 scans. GT when supervised +.2 head KD from own frozen r96 dense-spatial/deep-period3 teacher. No feature KD. Teacher only supplies loss on train split; no state injection. Both arms keep original own deep period3 cache; dense arm no spatial reuse, cache arm own compact period3 spatial recurrence, threshold.75/confidence.1.',
        recurrence='Differentiable masked_dense computes all convolutions during training but writes only selected cells. Old own spatial/FM/deep states detached each frame, no BPTT. Discrete masks not learned. All BN eval. Teacher and student keep independent histories. Sparse training speedup not claimed.',
        evaluation='Complete val5794/1738 labels at epoch0,1,2. Fixed dense/selected controls; trained full/dense/selected/masked_dense. Verify actual masks and same-input replay on disagreement; report AP, actual area/FLOPs, mean/P95/state bytes.',
        selection='Preserve every checkpoint and e0; best selected AP is exploratory on validation, not held-out generalization. Always report matching actual area/FLOPs and dense AP. Stop2epochs budget, not convergence; reassess before extension.')
    write_json(out/'contract.json',contract);write_json(out/'provenance.json',provenance)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    costs,gate=engineering(base,student,split,repair,cfg,train);write_json(out/'engineering.json',gate)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-4)
    teacher=SplitReadoutDetector(base,cfg,split,3,repair)
    if a.arm=='cache':mode=enable_cache_training(SpatialBasisCacheDetector(student,cfg,split,repair,.75,execution='masked_dense',confidence_threshold=.1))
    else:mode=SplitReadoutDetector(student,cfg,split,3,repair)
    best=-1.;history=[];frozen={n:p.detach().clone() for n,p in stage.named_parameters() if not p.requires_grad}
    buffers={n:b.detach().clone() for n,b in student.named_buffers()}
    for epoch in range(3):
        losses=[];areas=[]
        if epoch:
            ts=ss=None;_,stream=loader(train,cfg,4,303+epoch)
            for index,sample in enumerate(stream):
                ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
                with torch.no_grad():target,ts,_=teacher.step(ego,ts,meta)
                optimizer.zero_grad(set_to_none=True);pred,ss,info=training_step(mode,ego,ss,meta)
                loss=.2*foreground_head_loss(pred,target)
                if meta['supervised']:loss=loss+criterion(pred,ego['label_dict'])
                assert torch.isfinite(loss);loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,5.));optimizer.step()
                losses.append(float(loss.detach()));areas.append(info.get('spatial_write_fraction',1.))
                if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='train',epoch=epoch,scans=index+1,loss=float(np.mean(losses)),mean_write=float(np.mean(areas))))
            assert index+1==len(train)
        assert all(torch.equal(frozen[n],p) for n,p in stage.named_parameters() if n in frozen)
        assert all(torch.equal(buffers[n],b) for n,b in student.named_buffers())
        assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
        directory=out/('epoch_%03d'%epoch);directory.mkdir();path=directory/'stage.pth'
        torch.save(dict(stage=stage.state_dict(),contract=contract,epoch=epoch),str(path))
        report=evaluate(base,student,split,repair,cfg,val,directory,costs,epoch)
        ap=report['results']['selected']['ap70']
        if ap>best:
            best=ap;write_json(out/'best.json',dict(epoch=epoch,ap70=ap,checkpoint=str(path),sha256=digest(path),metrics=report['results']['selected']))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,train_write=float(np.mean(areas)) if areas else None,validation=report))
        write_json(out/'history.json',history)
        torch.save(dict(stage=stage.state_dict(),optimizer=optimizer.state_dict(),contract=contract,epoch=epoch,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()),str(out/'latest.tmp'))
        os.replace(str(out/'latest.tmp'),str(out/'latest.pth'))
    write_json(out/'report.json',dict(status='COMPLETED',epochs=2,stop_reason='diagnostic_budget_not_convergence',best=json.loads((out/'best.json').read_text())))
    write_json(out/'status.json',dict(status='COMPLETED',epochs=2))


def main():
    p=argparse.ArgumentParser()
    for key in ('root','readout-run','repair-run','memory-run','projected-run','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--arm',choices=('dense','cache'),required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Cache adaptation exceeded12hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(43200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.train_int_spatial_cache_adapt']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=43200,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
