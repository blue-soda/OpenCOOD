"""Instrument existing fixed-checkpoint sparse pilots; no policy or weight search."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import opencood.models.sub_modules.int_unified_bev as dense_module
import opencood.models.sub_modules.int_unified_sparse as sparse_module
import opencood.models.sub_modules.int_unified_gather as gather_module
import opencood.models.sub_modules.int_cache_repair as repair_module
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.models.sub_modules.int_unified_sparse import SelectiveUnifiedBEV
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


class Timers:
    def __init__(self):self.restore=[];self.events=[]

    def wrap(self,obj,attribute,name):
        original=getattr(obj,attribute);self.restore.append((obj,attribute,original))
        def measured(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();tick=time.perf_counter()
            result=original(*args,**kwargs)
            cpu=1000*(time.perf_counter()-tick);end.record()
            self.events.append((name,start,end,cpu));return result
        setattr(obj,attribute,measured)

    def close(self):
        for obj,attribute,original in reversed(self.restore):setattr(obj,attribute,original)

    def collect(self):
        result={}
        for name,start,end,cpu in self.events:
            row=result.setdefault(name,dict(gpu_timeline_ms=0.,cpu_dispatch_ms=0.,calls=0))
            row['gpu_timeline_ms']+=start.elapsed_time(end);row['cpu_dispatch_ms']+=cpu;row['calls']+=1
        self.events=[];return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pilot-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=256)
    p.add_argument('--pilot16',default='unified_sparse_e1_t16')
    p.add_argument('--pilot32',default='unified_sparse_e1_t32');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:run(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


def run(a,out):
    for k,v in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0).items():setattr(a,k,v)
    source=Path(a.pilot_root);contracts={}
    for tile in [16,32]:
        pilot=source/(a.pilot16 if tile==16 else a.pilot32)
        assert json.loads((pilot/'status.json').read_text())['status']=='COMPLETED'
        contracts[tile]=json.loads((pilot/'experiment_contract.json').read_text())
    contract=contracts[16];assert contract['checkpoint_sha256']==contracts[32]['checkpoint_sha256']==digest(Path(contract['checkpoint']))
    executor=contract.get('executor','sparse')
    assert executor==contracts[32].get('executor','sparse')
    teacher,cfg,rows,provenance=setup(a)
    student=UnifiedBEVDetector(teacher,cfg,'v2','recurrent').cuda().eval()
    saved=torch.load(contract['checkpoint'],map_location='cpu')
    assert saved['contract']['version']=='unified_bev_dense_v2'
    assert saved['contract']['data_contract']==provenance['contract']
    student.load_state_dict(saved['model'],strict=True)
    write_json(out/'provenance.json',dict(source_contracts=contracts,source_sha256=contract['checkpoint_sha256'],setup=provenance))
    reports={}
    for name,tile,mode in [('dense_before',None,None),(executor+'16',16,executor),(executor+'32',32,executor),
                           ('masked32',32,'masked_dense'),('dense_after',None,None)]:
        write_json(out/'status.json',dict(status='RUNNING',phase=name))
        wrapper=student if tile is None else SelectiveUnifiedBEV(student,contracts[tile]['calibration']['thresholds']['q75'],
            tile,contracts[tile]['max_cell_age'],mode,layout=contracts[tile].get('layout','merged'),
            batch_size=contracts[tile].get('crop_batch',8))
        timers=Timers()
        for obj,attribute,label in [(student,'spatial','pillar'),(student.core,'evidence','evidence_total'),
                (student.core.observation,'forward','observation'),(student.core.context,'forward','context'),
                (student.core.update,'forward','update_calls'),(student.core,'predict','readout_heads'),
                (dense_module,'warp_bev','warp'),(sparse_module,'warp_bev','warp'),
                (sparse_module,'change_scores','change_score'),(sparse_module,'refresh_tiles','crop_execution_total'),
                (repair_module,'rectangles','rectangle_plan'),(gather_module,'gathered_update','gather_compute')]:
            timers.wrap(obj,attribute,label)
        state=None;records=[];_,stream=loader(rows[:a.frames],cfg,4,303)
        try:
            with torch.no_grad(),(out/(name+'_frames.jsonl')).open('x',buffering=1) as log:
                for index,sample in enumerate(stream):
                    ego=to_device(sample['ego'],torch.device('cuda'));timers.events=[]
                    torch.cuda.synchronize();tick=time.perf_counter()
                    prediction,state,info=wrapper.step(ego,state,sample['meta'])
                    torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
                    stages=timers.collect();assert all(torch.isfinite(v).all() for v in prediction.values())
                    row=dict(frame=sample['meta']['frame'],ms=elapsed,stages=stages,update_stats=info.get('update_stats'))
                    log.write(json.dumps(row)+'\n')
                    if index>=10:records.append(row)
        finally:timers.close()
        keys=sorted({k for r in records for k in r['stages']})
        report=dict(frames=a.frames,measured_frames=len(records),mean_ms=float(np.mean([x['ms'] for x in records])),
            stages={k:{m:float(np.mean([x['stages'].get(k,{}).get(m,0.) for x in records])) for m in
                      ['gpu_timeline_ms','cpu_dispatch_ms','calls']} for k in keys})
        reports[name]=report;print(name,json.dumps(report),flush=True)
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],executor=executor,reports=reports,
        caveats=['Instrumented prefix at early checkpoint, not an uninstrumented speed benchmark or AP evaluation.',
                 'CUDA-event spans may include host-induced GPU idle; CPU dispatch is asynchronous and not kernel time.',
                 'Nested stages overlap: evidence includes observation/context; crop_execution includes update/gather/rectangle_plan. Do not sum all columns.',
                 'Selector masks, concatenation and cache writeback outside named spans remain in total wall time. Other GPUs are training.']))
    write_json(out/'status.json',dict(status='COMPLETED'))


if __name__=='__main__':main()
