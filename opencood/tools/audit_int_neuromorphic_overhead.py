"""Conditional memory/routing break-even, not measured device energy or latency.

Uses the SAME R091 quantized representation for event and sparse-count paths.
All unit costs are symbolic ratios to one MAC. No vendor energy constants.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    source=root/'spatial_event_audit_20261101/final_verification.json'
    raw=source.read_bytes();prior=json.loads(raw)
    assert prior['status']=='VERIFIED' and prior['scans']==5794
    n=prior['scans'];totals=prior['totals'];base=prior['theory']
    ac=sum(v['event_ac_exact'] for v in totals.values())/n
    count=sum(v['sparse_count_mac_exact'] for v in totals.values())/n
    packets=sum(v['spike_count'] for v in totals.values())/n
    active=sum(v['active_count_elements'] for v in totals.values())/n
    assert math.isclose(ac,base['event_ac_per_frame'],rel_tol=1e-12)
    assert math.isclose(count,base['sparse_count_mac_per_frame'],rel_tol=1e-12)
    assert math.isclose(count/ac,base['ac_over_mac_energy_break_even_against_sparse_count'],rel_tol=1e-12)
    assert active <= packets <= 8*active and ac > count
    # Hypothesis: one 32-bit weight read per valid synaptic delivery, both arms.
    # Logical on-chip requests, NOT external-memory bytes or unique weights.
    excess_reads=ac-count
    event_bytes=4*packets  # 32-bit linear source address; time slot implicit
    payloads={}
    for packet_size in (5,8):
        count_bytes=packet_size*active
        payloads[str(packet_size)]=dict(count_packet_bytes=packet_size,
            event_injection_bytes_per_frame=event_bytes,
            count_injection_bytes_per_frame=count_bytes,
            event_minus_count_injection_bytes_per_frame=event_bytes-count_bytes)
    sensitivities=[]
    for reuse_factor in (1.,.25):
        for beta in (0.,.1,.5,1.):
            difference=excess_reads*reuse_factor
            threshold=(count-beta*difference)/ac
            margin=count-.1*ac-beta*difference
            sensitivities.append(dict(event_and_count_read_fraction=reuse_factor,
                e_read_over_e_mac=beta,ac_over_mac_break_even=threshold,
                remaining_extra_overhead_budget_at_r01_mac_energy_units_per_frame=margin,
                event_wins_before_other_cost_at_r01=margin>0))
    beta_limit=(count-.1*ac)/excess_reads
    return dict(status='VERIFIED_CONDITIONAL_ANALYSIS',run='R-095',scans=n,
        source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),
        quantities=dict(event_ac_per_frame=ac,sparse_count_mac_per_frame=count,
            event_source_packets_per_frame=packets,nonzero_count_source_packets_per_frame=active,
            extra_logical_weight_reads_per_frame=excess_reads,
            event_logical_fp32_weight_read_bytes_per_frame=4*ac,
            count_logical_fp32_weight_read_bytes_per_frame=4*count),
        injection_payloads=payloads,sensitivities=sensitivities,
        e_read_over_e_mac_limit_at_r01_unit_read_fractions=beta_limit,
        general_margin='C - r*A - beta*(rho_event*A-rho_count*C) - gamma*(B_event-B_count) - delta_other',
        units='A,C: per-frame deliveries; beta=e_read/eMAC; gamma=e_injection_byte/eMAC; delta_other in eMAC/frame. Positive margin favors events.',
        assumptions=['Same frozen R091 encoded values and analog exceptions; unchanged ANN/analog costs cancel in the difference.',
            'FP32 weight logical reads counted per valid delivery before stated reuse fraction; no claim about actual DRAM/SRAM traffic.',
            'Event packet:32-bit source index, implicit T8 time slot, no header. Count packet:32-bit index plus count byte, either5 bytes packed or8 aligned. Both omit routing headers, hops and multicast replication.',
            'Both per-read costs assumed equal after choosing bit width; r and beta are hypothetical sensitivity parameters, not hardware specifications.'],
        limitations=['Encoding, neuron updates, accumulator memory, barriers, other traffic and static-power differences remain in delta_other; not assumed zero outside explicit scenario.',
            'Packets injected at source differ from delivered synapses; packed vs aligned packet assumptions can reverse byte comparisons.',
            'Small-integer count multiplication may be cheaper than a general MAC; this reference is not guaranteed optimal low-bit hardware.',
            'R093 FP32 rearrangement slightly changes codes; this SAME-representation comparison is based on R091, not exact R093 device operations.',
            'No device milliseconds, joules or total-energy reduction established.'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.evidence_root)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('quantities','injection_payloads','sensitivities','e_read_over_e_mac_limit_at_r01_unit_read_fractions')},indent=2))


if __name__=='__main__':main()
