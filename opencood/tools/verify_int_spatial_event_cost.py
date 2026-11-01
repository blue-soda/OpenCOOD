"""Verify R091 immutable evidence and compare two hypothetical sparse backends.

Standard-library CPU audit. The sparse count-MAC comparator uses the same
quantized representation as the event implementation. Neither is a GPU kernel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    hashes = {}

    def read(path, lines=False):
        raw = (root/path).read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    new = 'spatial_event_audit_20261101/final_v1/readout_spatial_cost_full_v1/'
    old = 'pyramid_cache_20261101/readout_tail_centered_full_v1/'
    report, reference = read(new+'report.json'), read(old+'report.json')
    frames, gold = read(new+'frames.jsonl', True), read(old+'frames.jsonl', True)
    n = len(frames)
    assert report['status'] == reference['status'] == 'COMPLETED'
    assert n == len(gold) == report['scans'] == reference['scans'] == 5794
    assert report['labels'] == reference['labels'] == 1738
    assert report['metrics'] == reference['metrics']
    assert read(new+'ap_records.json') == read(old+'ap_records.json')
    assert report['checkpoint_sha256'] == reference['checkpoint_sha256']
    assert report['scale_sha256'] == reference['scale_sha256']
    assert report['parameters'] == 32756052
    assert report['timing_valid'] is False and report['model_step_ms_mean'] is None
    totals = {site: dict.fromkeys(('event_ac_exact', 'event_ac_upper', 'analog_mac_exact',
              'analog_mac_upper', 'sparse_count_mac_exact', 'spike_count', 'active_count_elements'), 0)
              for site in ('input', 'middle')}
    for index, (a, b) in enumerate(zip(frames, gold)):
        for key in ('frame', 'segment', 'supervised', 'reset', 'late_refresh', 'anchor_timestamp_us',
                    'core_conv_flops', 'pillar_linear_flops', 'cache_bytes'):
            assert a[key] == b[key], (index, key)
        assert a['validation_if_count_exact'] == (True if index < 32 else None)
        for site in totals:
            coding = a['activation_coding'][site]
            assert all(coding[k] == v for k, v in b['activation_coding'][site].items())
            cost = coding['spatial_cost']
            assert cost['shape'] == [1, 384 if site == 'input' else 256, 100, 252]
            assert cost['out_channels'] == 256
            assert cost['spike_count'] == coding['spike_count']
            assert cost['analog_bypass_elements'] == coding['analog_bypass_elements']
            assert cost['event_ac_upper'] == cost['spike_count']*2304
            assert cost['analog_mac_upper'] == cost['analog_bypass_elements']*2304
            assert cost['spike_count']*1024 <= cost['event_ac_exact'] <= cost['event_ac_upper']
            assert cost['analog_bypass_elements']*1024 <= cost['analog_mac_exact'] <= cost['analog_mac_upper']
            assert cost['active_count_elements'] <= cost['spike_count'] <= 8*cost['active_count_elements']
            assert cost['sparse_count_mac_exact'] <= cost['event_ac_exact'] <= 8*cost['sparse_count_mac_exact']
            assert all(cost[k] % 256 == 0 for k in ('event_ac_exact', 'analog_mac_exact', 'sparse_count_mac_exact'))
            for key in totals[site]:
                totals[site][key] += cost[key]
    for site, values in totals.items():
        assert values == report['sites'][site]['spatial_totals']
        assert all(math.isclose(v/n, report['sites'][site]['spatial_mean_per_frame'][k], rel_tol=1e-12)
                   for k,v in values.items())
    core = sum(f['core_conv_flops'] for f in frames)/n
    assert math.isclose(core, report['core_conv_flops_mean'], rel_tol=1e-12)
    budget = read('metrics_neuromorphic_20261101/report.json')
    ann_mac = core/2
    readout_mac = budget['theory']['readout_ann_mac_per_frame']
    remaining_mac = ann_mac-readout_mac
    ac = sum(v['event_ac_exact'] for v in totals.values())/n
    ac_upper = sum(v['event_ac_upper'] for v in totals.values())/n
    tail = sum(v['analog_mac_exact'] for v in totals.values())/n
    sparse = sum(v['sparse_count_mac_exact'] for v in totals.values())/n
    sparse_core = remaining_mac+tail+sparse
    event_mac = remaining_mac+tail
    sensitivities = []
    for ratio in (1., .5, .1):
        event_energy = event_mac+ratio*ac
        sensitivities.append(dict(e_ac_over_e_mac=ratio,
            event_synaptic_energy_ratio_to_dense=event_energy/ann_mac,
            event_synaptic_energy_ratio_to_sparse_count=event_energy/sparse_core,
            extra_overhead_break_even_mac_energy_units_per_frame=sparse-ratio*ac))
    return dict(status='VERIFIED',run='R-091',source_sha256=hashes,scans=n,labels=1738,
        raw_ap_records_equal=True,original_coding_core_cache_equal=True,explicit_if_parity_frames=32,
        measured=dict(parameters=report['parameters'],core_conv_flops_per_frame=core,
            pillar_linear_flops_per_frame=report['pillar_linear_flops_mean'],model_mean_ms=None,
            mean_ms_missing_reason=report['timing_invalid_reason'],metrics=report['metrics']),
        totals=totals,theory=dict(dense_core_mac_per_frame=ann_mac,remaining_core_mac_per_frame=remaining_mac,
            event_ac_per_frame=ac,analog_tail_mac_per_frame=tail,sparse_count_mac_per_frame=sparse,
            hybrid_core_mac_per_frame=event_mac,sparse_count_core_mac_per_frame=sparse_core,
            boundary_ac_reduction_fraction=1-ac/ac_upper,
            sparse_count_synaptic_energy_ratio_to_dense=sparse_core/ann_mac,
            ac_over_mac_energy_break_even_against_sparse_count=sparse/ac,sensitivities=sensitivities),
        limitations=['Both event AC and sparse-count MAC are hypothetical execution with identical encoded values and zero-padding treatment.',
            'Core convolution synaptic dynamic energy only; unequal encoding/routing/memory/neuron/static overhead must be added.',
            'No sparse kernel, finite-precision device AP, hardware milliseconds or joules measured.',
            'Single seed validation-selected weights, not independent test or statistical superiority.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = verify(args.evidence_root)
    result['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'],scans=result['scans'],theory=result['theory']),indent=2))


if __name__ == '__main__':
    main()
