"""Frozen-trace coverage and optimistic sparse-ANN headroom, without torch.

No candidate AP, event rate, hardware joules or latency is predicted. Nonzero
locations were not retained in R106, so sparse deliveries are bounded, not exact.
"""
import argparse
from collections import Counter
from functools import lru_cache, reduce
import hashlib
import json
from pathlib import Path
from operator import mul


def prod(values):
    return reduce(mul, values, 1)


@lru_cache(None)
def fanout_histogram(kind, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, channels):
    axes=[]
    for size, output, kernel, stride, pad, dilation in [(ih,oh,kh,sh,ph,dh),(iw,ow,kw,sw,pw,dw)]:
        counts=[0]*size
        if kind=='Conv2d':
            for y in range(output):
                for k in range(kernel):
                    x=y*stride-pad+k*dilation
                    if 0<=x<size:counts[x]+=1
        else:
            assert kind=='ConvTranspose2d'
            for x in range(size):
                counts[x]=sum(0<=x*stride-pad+k*dilation<output for k in range(kernel))
        axes.append(Counter(counts))
    result=Counter()
    for fy,ny in axes[0].items():
        for fx,nx in axes[1].items():result[fy*fx]+=ny*nx*channels
    return tuple(sorted(result.items()))


def extrema(histogram, nonzero):
    assert 0<=nonzero<=sum(n for _,n in histogram)
    result=[]
    for entries in (histogram,tuple(reversed(histogram))):
        remain=nonzero;value=0
        for fanout,capacity in entries:
            take=min(remain,capacity);value+=take*fanout;remain-=take
        assert remain==0;result.append(value)
    return result


def audit(a):
    source=Path(a.operator_audit);full=Path(a.full_reference)
    inventory=json.loads((source/'inventory.json').read_text());report=json.loads((source/'report.json').read_text())
    frames=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()]
    gold=[json.loads(x) for x in (full/'full_frames.jsonl').read_text().splitlines()]
    fr=json.loads((full/'report.json').read_text())
    assert report['status']==fr['status']=='COMPLETED' and len(frames)==256 and len(gold)==5794
    assert report['checkpoint_sha256']==fr['checkpoint_sha256']
    observed={};prefix=sum(f['core_conv_flops'] for f in frames)/len(frames)
    full_flops=fr['evaluation']['metrics']['strict']['core_conv_flops_mean']
    refresh=sum(f['arms']['strict']['refresh'] is not None for f in gold)
    for frame in frames:
        assert sum(c['core_conv_flops'] for c in frame['calls'])==frame['core_conv_flops']
        for c in frame['calls']:
            name=c['name'];m=inventory['convolutions'][name]
            n,ic,ih,iw=c['input_shape'];_,oc,oh,ow=c['output_shape']
            h=fanout_histogram(m['type'],ih,iw,oh,ow,*m['kernel'],*m['stride'],*m['padding'],*m['dilation'],n*ic)
            low,high=extrema(h,c['input_nonzero']);low*=oc//m['groups'];high*=oc//m['groups']
            v=observed.setdefault(name,dict(calls=0,flops=0,nonzero=0,elements=0,low=0,high=0,shape=c['input_shape'],per_call_flops=c['core_conv_flops']))
            assert v['shape']==c['input_shape'] and v['per_call_flops']==c['core_conv_flops']
            v['calls']+=1;v['flops']+=c['core_conv_flops'];v['nonzero']+=c['input_nonzero'];v['elements']+=c['input_elements'];v['low']+=low;v['high']+=high
    # Validate the full scheduled denominator from per-call geometry and actual
    # refresh count, rather than transplanting the prefix deep-refresh frequency.
    scheduled={n:v['per_call_flops']*(refresh if n.startswith(('deep_encoder.','deep_update.')) else len(gold)) for n,v in observed.items()}
    assert sum(scheduled.values())/len(gold)==full_flops
    params=sum(p['elements'] for p in inventory['parameters']);assert params==32756052
    base_bytes=sum(v['bytes'] for v in frames[0]['cache'].values());assert base_bytes==22579200
    definitions=[]
    def add(name,targets,groups=None,replace=False):
        groups=groups or {t:t for t in targets};definitions.append((name,targets,groups,replace))
    add('coarse_last1',['coarse_encoder.3.conv2'])
    add('coarse_last2',['coarse_encoder.2.conv2','coarse_encoder.3.conv2'])
    add('coarse_all4_conv2',['coarse_encoder.%d.conv2'%i for i in range(4)])
    coarse=[n for n in observed if n.startswith('coarse_encoder.')]
    add('coarse_all_conv',coarse,{t:('coarse_first_input' if t in ('coarse_encoder.0.conv1','coarse_encoder.0.downsample.0') else t) for t in coarse})
    add('readout_last',['readout.layers.0.double_conv.3'])
    add('readout_both',['readout.layers.0.double_conv.0','readout.layers.0.double_conv.3'])
    consumers={'fine':['coarse_encoder.0.conv1','coarse_encoder.0.downsample.0','deblocks.0.0'],
        'coarse':['deep_encoder.0.conv1','deep_encoder.0.downsample.0','deblocks.1.0'],'deep_anchor':['deblocks.2.0']}
    for stages in [('fine',),('coarse',),('fine','coarse','deep_anchor')]:
        targets=[t for stage in stages for t in consumers[stage]]
        add('existing_state_'+('_'.join(stages)),targets,{t:stage for stage in stages for t in consumers[stage]},True)
    result=[]
    for name,targets,groups,replace in definitions:
        memories={}
        for t in targets:
            shape=observed[t]['shape'];key=groups[t]
            if key in memories:assert memories[key]==shape
            memories[key]=shape
        elements=sum(prod(s) for s in memories.values());target=sum(observed[t]['flops'] for t in targets)/len(frames)
        low=sum(observed[t]['low'] for t in targets)/len(frames);high=sum(observed[t]['high'] for t in targets)/len(frames)
        rest=(prefix-target)/2
        result.append(dict(name=name,targets=targets,memories=memories,state_replacement_hypothesis=replace,
            total_parameters_if_all_original_operators_retained=params,target_weight_parameters=sum(prod(inventory['convolutions'][t]['weight_shape']) for t in targets),
            prefix_scans=len(frames),prefix_target_flops=target,prefix_dense_coverage_percent=100*target/prefix,
            full_target_flops_from_schedule=sum(scheduled[t] for t in targets)/len(gold),
            full_dense_coverage_percent=100*sum(scheduled[t] for t in targets)/len(gold)/full_flops,
            source_nonzero_fraction={t:observed[t]['nonzero']/observed[t]['elements'] for t in targets},
            prefix_sparse_ann_macs_interval=[low,high],
            prefix_free_AC_extra_core_energy_saving_percent_interval=[100*low/(rest+low),100*high/(rest+high)],
            persistent_fp32_total_bytes=base_bytes+(0 if replace else 4*elements),
            persistent_fp16_total_bytes=base_bytes+(-2*elements if replace else 2*elements),
            distinct_membrane_elements=elements,scale_fp32_bytes=sum(s[1] for s in memories.values())*4,
            candidate_ap50=None,candidate_ap70=None,candidate_model_ms=None,
            caveat='Frozen prefix bounds; zero-cost events and all overhead free; changed activations after adaptation unknown. State replacement requires a new validated representation; FP16 AP/quantization/warp implementation untested.'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(status='DERIVED',run='R-115',prefix_scans=len(frames),prefix_core_flops=prefix,
        full_scans=len(gold),full_refreshes=refresh,full_core_flops=full_flops,parameters=params,baseline_cache_bytes=base_bytes,candidates=result,
        baseline_metrics_source=dict(prefix=report['metrics'],full=fr['evaluation']['metrics']['strict']),
        scope='Only original core Conv; candidates not implemented or evaluated. Prefix nonzero placement bounds are not full-val sparse rates. Coverage is not energy saving; no hypothetical ms or AP.',
        source_sha256={str(p):sha(p) for p in [source/'inventory.json',source/'frames.jsonl',source/'report.json',full/'full_frames.jsonl',full/'report.json']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--operator-audit',required=True);p.add_argument('--full-reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    for c in result['candidates']:
        print(c['name'], 'full coverage %.4f%%'%c['full_dense_coverage_percent'], 'freeAC prefix bound',c['prefix_free_AC_extra_core_energy_saving_percent_interval'],
              'state32/16 MB',c['persistent_fp32_total_bytes']/1e6,c['persistent_fp16_total_bytes']/1e6)


if __name__=='__main__':main()
