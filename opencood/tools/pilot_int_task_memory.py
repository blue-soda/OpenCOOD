"""Matched fixed-budget task-memory adaptation versus original/reset controls."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_task_membrane_detector import TaskMembraneDetector
from opencood.tools.pilot_int_gated_state import train_one,evaluate,expected_updates
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());v=json.loads(Path(a.verification).read_text());assert audit['status']==v['status']=='VERIFIED' and v['run']=='R-144'
    assert digest(Path(a.checkpoint))==v['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'];base.load_state_dict(saved['model']);del teacher,saved
    for p in base.parameters():p.requires_grad_(False)
    models={n:TemporalPyramidDetector(base,'original').cuda().eval() for n in ('frozen','original')}
    models.update({n:TaskMembraneDetector(base,n).cuda().eval() for n in ('continuous','reset')})
    for n,m in models.items():
        for name,p in m.named_parameters():p.requires_grad_(n!='frozen' and name.startswith('core.coarse_update.'))
    source=models['original'].core.coarse_update.state_dict()
    for n in ('continuous','reset'):assert all(torch.equal(x,models[n].core.coarse_update.candidate.state_dict()[k]) for k,x in source.items())
    train,evaluation=rows[:512],rows[512:768];updates=expected_updates(train);assert len(updates)==71
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),train_frames=[r['frame'] for r in train],evaluation_frames=[r['frame'] for r in evaluation],expected_update_frames=updates,
        seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,tau=.2,loss='Same supervised-window mean detection only, no distillation; only coarse candidate trained; BN/everything else frozen.',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),boundary='Original/continuous/reset share initial candidate values and71 updates, not SNN training. Reset clears coarse state only; fine/deep ANN history remains. Fixed512/232 labels; disjoint train256 diagnostic. No fullval/convergence or SNN advantage claim.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();names=('original','continuous','reset')
    preflight={n:train_one(models[n],cfg,train,out,n,criterion,contract,True) for n in names};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial');learned={n:train_one(models[n],cfg,train,out,n,criterion,contract) for n in names};final,raw1=evaluate(models,base,cfg,evaluation,out,'final');assert raw0['frozen']==raw1['frozen']
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,trained=True,timing_valid=False,frozen_raw_ap_exact=True));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min matched task memory learning')));signal.alarm(5400)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
