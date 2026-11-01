"""Conditional integer delivery/shift/packet budget; not device energy."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frames',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.frames);rows=[json.loads(x) for x in source.read_text().splitlines()]
    h=[0]*9;w=[0]*9;merge=0
    for row in rows:
        for site in row['arms']['bitplane_int8_i32']['sites'].values():
            c=site['ideal_spatial_cost']
            h=[x+y for x,y in zip(h,c['count_histogram'])]
            w=[x+y for x,y in zip(w,c['valid_delivery_histogram'])]
            merge+=site['integer_bit_shifts']
    n=len(rows);pop=[bin(c).count('1') for c in range(9)]
    A=sum(c*v for c,v in enumerate(w))/n
    B=sum(c*v for c,v in zip(pop,w))/n
    C=sum(w[1:])/n
    D=sum(bin(c>>1).count('1')*v for c,v in enumerate(w))/n
    E=sum(c*v for c,v in enumerate(h))/n
    P=sum(c*v for c,v in zip(pop,h))/n
    assert C<=B<=A and 0<D<=B
    result=dict(status='COMPLETED',scans=n,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        delivery_per_frame=dict(unary_ac=A,bitplane_ac=B,sparse_count_mac=C,nonidentity_per_delivery_shifts=D),
        shared_ac_cost_margin_per_frame=A-B,
        per_delivery_shift_cost_break_even_eAC=(A-B)/D,
        shift_cost_model='Unary minus bitplane = (A-B)*eAC - D*eShift - delta_encoding_routing_memory_static. Positive favors bitplane.',
        alternative_plane_accumulators=dict(dense_output_merge_adds_per_frame=merge/n,output_shifts_per_frame=merge/n,
            margin_before_shift_and_memory_in_eAC=A-B-merge/n,
            boundary='Four plane accumulators then dense merge; separate alternative to per-delivery shift. Additional accumulator storage/traffic excluded, not free.'),
        source_packets_per_frame=dict(unary=E,bitplane=P),
        source_bytes_per_frame=dict(unary_address4=4*E,bitplane_address4_plus_bit_index1=5*P,bitplane_aligned8=8*P),
        boundary='Same INT8 q/count0..8, shared analog tail/scale/BN cancel. Logical valid fanout, no zero-weight skipping. Packet model 32-bit source address + 2-bit plane index in one byte or aligned8; implicit event timeslot, no headers/multicast/hops. Equal eAC is an assumption. Excludes neuron/encoding/memory/static; no measured joules or ms. Split belongs to source run.')
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
