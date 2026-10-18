"""Diagnostic stage spans; instrumentation is not a deployment speed result."""
import time
import torch


class RepairStageProfiler:
    def __init__(self, detector, model, repairs):
        self.detector=detector;self.saved=[];self.pending=[];self.records=[]
        targets=[('pillar',model.pillar_vfe,'forward'),('scatter',model.scatter,'forward'),
                 ('fm',model.feature_memory,'step'),
                 ('decode',model.backbone,'decode_multiscale_feature'),
                 ('classification',model.cls_head,'forward'),('regression',model.reg_head,'forward')]
        for i in range(3):targets.append(('stage%d'%(i+1),getattr(model.backbone.resnet,'layer%d'%i),'forward'))
        if model.shrink_flag:targets.append(('shrink',model.shrink_conv,'forward'))
        if model.use_dir:targets.append(('direction',model.dir_head,'forward'))
        for i,module in enumerate(repairs):targets.append(('repair_decoded' if len(repairs)==1 else 'repair%d'%(i+2),module,'forward'))
        for name,obj,attr in targets:
            original=getattr(obj,attr);self.saved.append((obj,attr,original))
            setattr(obj,attr,self.wrap(name,original))

    def wrap(self,name,original):
        def timed(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();start.record()
            result=original(*args,**kwargs)
            end.record();host=1000*(time.perf_counter()-tick)
            self.pending.append((name,start,end,host))
            return result
        return timed

    def step(self,data,state,meta):
        self.pending=[]
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        prediction,state,info=self.detector.step(data,state,meta)
        end.record();end.synchronize()
        stages={}
        for name,a,b,host in self.pending:
            item=stages.setdefault(name,dict(cuda_span_ms=0.,host_call_ms=0.,calls=0))
            item['cuda_span_ms']+=a.elapsed_time(b);item['host_call_ms']+=host;item['calls']+=1
        total=start.elapsed_time(end)
        record=dict(partial=bool(info['local_partial']),cuda_span_ms=total,stages=stages,
                    other_cuda_span_ms=total-sum(v['cuda_span_ms'] for v in stages.values()))
        self.records.append(record);info['stage_profile']=record
        return prediction,state,info

    def close(self):
        for obj,attr,original in reversed(self.saved):setattr(obj,attr,original)
        self.saved=[]

    def summary(self):
        measured=self.records[10:]
        result={}
        for phase,records in [('all',measured),('refresh',[r for r in measured if not r['partial']]),
                              ('repair',[r for r in measured if r['partial']])]:
            if not records:continue
            names=sorted({k for r in records for k in r['stages']})
            stages={name:{key:sum(r['stages'].get(name,{}).get(key,0) for r in records)/len(records)
                          for key in ['cuda_span_ms','host_call_ms','calls']} for name in names}
            result[phase]=dict(frames=len(records),stages=stages,
                cuda_span_ms=sum(r['cuda_span_ms'] for r in records)/len(records),
                other_cuda_span_ms=sum(r['other_cuda_span_ms'] for r in records)/len(records))
        return dict(groups=result,warmup_discarded=10,
            caveats=['Diagnostic prefix; not full-validation accuracy or production latency evidence.',
                     'Disjoint top-level CUDA event spans can include device idle time waiting for host launches; not pure kernel compute or utilization.',
                     'Host call durations include asynchronous launch and possible blocking; do not add host and CUDA spans.',
                     'Other span includes deep-cache alignment, input assembly, bookkeeping and instrumentation gaps.',
                     'No stage-level synchronization; one synchronization at model-step end. Instrumentation adds overhead.'])
