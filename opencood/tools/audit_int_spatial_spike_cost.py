"""Conditional energy headroom from full real-frame activation evidence; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.evidence);out=Path(a.output)
    if out.exists():raise FileExistsError(str(out))
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8-sig'))
    verified=read('full_final_verification.json');run=root/'full_final_v1/spatial_spike_full_v1'
    assert verified['status']=='VERIFIED' and verified['scans']==5794 and verified['labels']==1738
    for name,sha in verified['source_sha256'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==sha
    modes=('ann','lif','leaky');totals={n:dict(deliveries=0,sources=0,history=0,membrane_bytes=0) for n in modes};scans=0
    with (run/'full_frames.jsonl').open() as f:
        for line in f:
            row=json.loads(line);scans+=1
            for n in modes:
                arm=row['arms'][n];counts=arm['output_support_4_6_9'];t=totals[n]
                t['deliveries']+=sum(k*v for k,v in zip((4,6,9),counts))*256;t['sources']+=sum(counts)
                if n!='ann':
                    t['history']+=int(arm['memory']['warp']);assert arm['memory']['state_bytes']==6451200
                    t['membrane_bytes']+=6451200*(1+int(arm['memory']['warp']))
    assert scans==5794
    for n,t in totals.items():
        t['deliveries_mean']=t['deliveries']/scans;t['sources_mean']=t['sources']/scans;t['membrane_scheduled_bytes_mean']=t['membrane_bytes']/scans
        t['source_packet_bytes_mean']=t['sources_mean']*(4 if n=='lif' else 8)
        assert t['deliveries_mean']==verified['theory'][n]['effective_deliveries_mean']
    assert totals['lif']['history']==totals['leaky']['history']==5580
    dense=verified['metrics']['strict']['core_conv_flops_mean']/2;target=3715891200.;rest=dense-target
    A=totals['lif']['deliveries_mean'];C=totals['ann']['deliveries_mean'];L=totals['leaky']['deliveries_mean'];M=totals['lif']['membrane_scheduled_bytes_mean']
    sensitivity=[]
    for ratio in (1.,.5,.1):
        event=rest+ratio*A;sparse_ann=rest+C;continuous=rest+L
        sensitivity.append(dict(ac_to_mac_energy_ratio=ratio,
            core_synapse_event_vs_dense_ann_saving_percent=100*(1-event/dense),
            core_synapse_sparse_ann_vs_dense_ann_saving_percent=100*(1-sparse_ann/dense),
            core_synapse_event_vs_sparse_ann_saving_percent=100*(1-event/sparse_ann),
            core_synapse_event_vs_leaky_saving_percent=100*(1-event/continuous),
            target_headroom_vs_ann_eMAC=C-ratio*A,target_headroom_vs_leaky_eMAC=L-ratio*A,
            memory_byte_cost_break_even_vs_ann_eMAC_per_byte=(C-ratio*A)/M,
            headroom_per_dense_neuron_update_vs_ann_eMAC=(C-ratio*A)/1612800))
    resource=dict(neurons_per_scan=1612800,history_neurons_mean=1612800*5580/scans,
        input_scale_multiplications_per_scan=1612800,decay_multiplications_mean=1612800*5580/scans,
        history_additions_mean=1612800*5580/scans,lif_comparisons_per_scan=1612800,lif_reset_subtractions_dense_per_scan=1612800,
        scalar_decay_evaluations_per_scan=1,extra_membrane_capacity_bytes=6451200,scale_buffer_bytes=1024,
        target_fp32_weights_bytes=589824*4,cache_increase_percent=100*6451200/22579200)
    result=dict(status='DERIVED_AND_CHECKED',run='R-110',source_run='R-109',scans=scans,labels=1738,
        metrics=verified['metrics'],streams=totals,core=dict(dense_macs_mean=dense,target_dense_macs=target,residual_dense_macs_mean=rest,
            target_dense_share_percent=100*target/dense),resource_schedule=resource,sensitivity=sensitivity,
        comparison_formula='C-r*A-beta*(rho_LIF*A-rho_baseline*C)-gamma_route*(B_LIF-B_baseline)-gamma_mem*(M_LIF-M_baseline)-delta_neuron_other > 0; replace C by baseline deliveries.',
        assumptions=['r=eAC/eMAC; beta=one weight-read energy/eMAC; rho=reads per delivery; gamma_route and gamma_mem are separate energy/byte normalized by eMAC; all hypothetical.',
            'Only target convolution is sparse/event; all other core convolutions remain dense at identical operation counts. Nonconvolution costs are excluded from core-only percentages.',
            'FP32 weight precision and resident capacity identical; scale folded offline into LIF/leaky weights replaces bank, no mandatory duplicate weight bank. Numerical folding equivalence remains unverified.',
            'Source packets:4B implicit-amplitude event address versus8B continuous address+FP32 value; exclude headers, multicast copies, hops and congestion.',
            'Membrane schedule:full FP32 write every frame, full read on history frames, fused nearest warp/neuron, no intermediate buffers. This is a conditional logical schedule, not a physical traffic lower bound.',
            'Break-even byte table sets weight/routing/neuron/all other differences to zero; not measured affordability. Shared membrane schedule cancels only against leaky under matched implementation.',
            'Dense neuron arithmetic includes silent cells; pruning invalid/zero cells requires its own matched sparse ANN policy. GPU implementation still dense. No joules or hardware milliseconds inferred.'],
        boundaries=['AP70 LIF-leaky difference is tiny at one seed; AP50 favors resets; no independent accuracy advantage or significance.',
            'Weight scaling/finite precision/device accumulation may change predictions. Current measured AP belongs to scaled-input GPU reference, not hypothetical hardware.'])
    result['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('full_final_verification.json',)}
    result['raw_source_sha256']=verified['source_sha256']
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(core=result['core'],streams=totals,resources=resource,sensitivity=sensitivity),indent=2))


if __name__=='__main__':main()
