"""CPU-only evidence projection and hypothetical neuromorphic synaptic budget.

No source evidence is modified. Output must be a new directory. FLOPs use
2 operations/MAC; AC counts are reported separately and are not GPU FLOPs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    sources = {}

    def read(name, lines=False):
        raw = (root / name).read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    prefix = 'pyramid_cache_20261101/'
    dense_path = 'pyramid_adaptation_20261031/protect_e6_final/protect_e6_replay_v2/report.json'
    strict_path = prefix + 'anchor_strict_final/anchor_strict_full_v1/report.json'
    patch_path = prefix + 'anchor_patch_full_final/anchor_patch_full_v2/report.json'
    hybrid_path = prefix + 'readout_tail_centered_full_v1/report.json'
    dense, strict, patch, hybrid = [read(p) for p in (dense_path, strict_path, patch_path, hybrid_path)]
    gate_path = 'pyramid_adaptation_20261031/gate/verification.json'
    gate = read(gate_path)['runs']['protect_gate_v1']
    full_timing_path = prefix + 'anchor_full_timing_final_verification.json'
    prefix_timing_path = prefix + 'anchor_patch_timing_final/anchor_patch_timing_v1/report.json'
    full_timing, prefix_timing = read(full_timing_path), read(prefix_timing_path)
    rows = []

    def row(name, report, params, flops, ms, note, refs):
        assert report['scans'] == 5794 and report['labels'] == 1738
        rows.append(dict(name=name, parameters=params, core_conv_flops_per_frame=flops,
                         model_mean_ms=ms, ap50_percent=100*report['metrics']['bev_ap50'],
                         ap70_percent=100*report['metrics']['bev_ap70'], scope=note, sources=refs))

    int_path = 'int_spike_results/spiking_gru_20261016/paired_report/report.json'
    int_params_path = 'int_spike_results/spiking_gru_20261016/implementation_audit.json'
    row('INT/GRU TBPTT4', read(int_path)['modes']['gru']['replay']['aligned'],
        read(int_params_path)['full_model_audit']['gru_total_parameters'], None, None,
        '该AP重放只有p50/p95；均时和统一核心FLOPs待补测', [int_path, int_params_path])
    row('R-051 新架构 dense', dense, gate['total'], gate['core_flops'],
        full_timing['timings']['dense']['mean_ms'], '均时R-074：全流5794帧×3轮',
        [dense_path, gate_path, full_timing_path])
    obs_path = 'observation_bank_20261101/final/three_frame_e1_replay_v2/report.json'
    obs_gate_path = 'observation_bank_20261031/gate/verification.json'
    obs = read(obs_gate_path)['observation_bank']
    row('R-062 前融合 AP支线', read(obs_path), obs['total_parameters'], obs['core_flops'], None,
        '核心量为8帧工程审计固定形状计数；有效均时未测', [obs_path, obs_gate_path])
    row('R-072 严格历史缓存', strict, gate['total'], strict['core_conv_flops_mean'],
        full_timing['timings']['period2']['mean_ms'], '均时R-074：全流5794帧×3轮',
        [strict_path, gate_path, full_timing_path])
    row('R-077 缓存+局部补丁', patch, gate['total'], patch['core_conv_flops_mean'],
        prefix_timing['timings']['patch']['mean_ms'], '均时R-078：仅512帧×4轮；勿与R-074直接比较',
        [patch_path, gate_path, prefix_timing_path])
    row('R-089 缓存+混合脉冲读出', hybrid, gate['total'], strict['core_conv_flops_mean'], None,
        'GPU仍密集Conv；hook与额外IF参考使均时无效；核心量逐帧核对strict',
        [hybrid_path, gate_path, strict_path])

    frames = read(prefix + 'readout_tail_centered_full_v1/frames.jsonl', True)
    gold = read(prefix + 'anchor_strict_final/anchor_strict_full_v1/frames.jsonl', True)
    n = len(frames)
    assert n == len(gold) == hybrid['scans'] == 5794
    for a, b in zip(frames, gold):
        for key in ('frame', 'segment', 'core_conv_flops', 'pillar_linear_flops', 'late_refresh', 'cache_bytes'):
            assert a[key] == b[key], (a['frame'], key)
    core = sum(f['core_conv_flops'] for f in frames)/n
    pillar = sum(f['pillar_linear_flops'] for f in frames)/n
    assert math.isclose(core, strict['core_conv_flops_mean'], rel_tol=1e-12)
    assert math.isclose(pillar, strict['pillar_linear_flops_mean'], rel_tol=1e-12)
    operators = read(prefix + 'operator_budget_v1/report.json')['operators']
    refresh = sum(f['late_refresh'] is not None for f in frames)
    stages = defaultdict(float)
    for name, op in operators.items():
        stage = name.split('.')[0]
        calls = refresh if stage in ('deep_encoder', 'deep_update') else n
        stages[stage] += op['flops']/op['calls']*calls/n
    assert math.isclose(sum(stages.values()), core, rel_tol=1e-12)
    sites = {}
    for site, op_name in [('input', 'readout.layers.0.double_conv.0'),
                          ('middle', 'readout.layers.0.double_conv.3')]:
        op = operators[op_name]
        assert op['kernel'] == [3, 3] and op['stride'] == [1, 1] and op['groups'] == 1
        values = [f['activation_coding'][site] for f in frames]
        assert all(v['steps'] == 8 and v['preserve_tail'] for v in values)
        totals = {key: sum(v[key] for v in values) for key in
                  ('elements', 'spike_count', 'analog_bypass_elements', 'analog_value_payload_bytes')}
        for key in ('elements', 'analog_bypass_elements'):
            assert totals[key] == hybrid['sites'][site][key]
        assert totals['spike_count'] <= 8*(totals['elements']-totals['analog_bypass_elements'])
        assert math.isclose(totals['spike_count']/(8*totals['elements']),
                            hybrid['sites'][site]['emission_rate'], rel_tol=1e-12)
        fanout = 9*op['out_channels']
        sites[site] = dict(totals, fanout_upper=fanout,
                           ac_upper_per_frame=totals['spike_count']*fanout/n,
                           analog_mac_upper_per_frame=totals['analog_bypass_elements']*fanout/n,
                           dense_flops_per_frame=op['flops']/op['calls'])
    readout = sum(v['dense_flops_per_frame'] for v in sites.values())
    ac = sum(v['ac_upper_per_frame'] for v in sites.values())
    tail_mac = sum(v['analog_mac_upper_per_frame'] for v in sites.values())
    remaining_mac = (core-readout)/2
    # Only the synaptic dynamic term, conditional on the stipulated hardware.
    sensitivity = [dict(e_ac_over_e_mac=r,
                        synaptic_energy_ratio_upper=(remaining_mac+tail_mac+r*ac)/(core/2))
                   for r in (1.0, 0.5, 0.1)]
    theory = dict(sites=sites, core_ann_mac_per_frame=core/2,
                  readout_ann_mac_per_frame=readout/2, remaining_ann_mac_per_frame=remaining_mac,
                  analog_tail_mac_upper_per_frame=tail_mac, event_ac_upper_per_frame=ac,
                  hybrid_mac_upper_per_frame=remaining_mac+tail_mac, sensitivity=sensitivity,
                  neuron_updates_if_dense_t8_per_frame=8*sum(v['elements'] for v in sites.values())/n,
                  analog_value_payload_bytes_per_frame=sum(v['analog_value_payload_bytes'] for v in sites.values())/n,
                  extra_event_weight_bank_fp32_bytes=sum(9*operators[k]['in_channels']*operators[k]['out_channels']*4
                      for k in ('readout.layers.0.double_conv.0','readout.layers.0.double_conv.3')),
                  scope='Hypothetical resident-weight event hardware; padding fanout upper bound; excludes neuron, encoding, routing, memory, BN/warp and static energy. Not measured GPU FLOPs, milliseconds or device energy.')
    return dict(status='VERIFIED', sources_sha256=sources, rows=rows, scans=n, labels=1738,
                core_flops_convention='2 FLOPs/MAC; Conv/ConvTranspose only; per scan including resets and refreshes',
                strict_module_flops_per_frame=dict(stages), pillar_linear_flops_per_frame=pillar,
                same_batch_prefix_timing=prefix_timing['timings'], theory=theory)


def markdown(report):
    lines = ['# 当前实验五项指标', '',
             'DAIR自车，seed303，完整val 5794扫描/1738标注，BEV AP（%）；验证选模，非独立测试。', '',
             '| 模型/策略 | 参数量 M | 核心卷积 GFLOPs/帧 | 模型均时 ms | AP50 % | AP70 % | 口径 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in report['rows']:
        flop = row['core_conv_flops_per_frame']
        ms = row['model_mean_ms']
        lines.append('| {} | {:.6f} | {} | {} | {:.4f} | {:.4f} | {} |'.format(
            row['name'], row['parameters']/1e6, '待补测' if flop is None else f'{flop/1e9:.3f}',
            '未测/无效' if ms is None else f'{ms:.4f}', row['ap50_percent'], row['ap70_percent'], row['scope']))
    lines += ['', '参数量为全部模型参数（含冻结参数），不含缓存与编码buffer。1 MAC=2 FLOPs；核心列不含Pillar Linear、BN、激活、warp和搬运，不能称整网全部操作量。strict/patch/hybrid另有Pillar Linear均值{:.6f} GFLOPs/帧。'.format(report['pillar_linear_flops_per_frame']/1e9),
              '', '均时均指模型，不含IO/体素化/后处理。R-074每臂17334个去预热样本；R-078每臂1984个样本。AP与独立计时来源分开保存，不用总耗时/帧数或p50替代均时。', '',
              'R-078同批前缀均时：'+ '，'.join('{}={:.4f} ms'.format(k, report['same_batch_prefix_timing'][k]['mean_ms']) for k in ('dense','period2','patch'))+'。', '',
              '## strict完整流模块计算量', '', '| 模块 | GFLOPs/帧 |', '|---|---:|']
    lines += ['| {} | {:.6f} |'.format(k,v/1e9) for k,v in report['strict_module_flops_per_frame'].items()]
    lines += ['', '模块值由R-084各算子每次执行量乘R-089完整流实际刷新次数推导，合计逐帧匹配R-072核心计数；不是把训练前缀比例直接外推。', '',
              '字段来源与SHA256见同目录report.json；本表为证据投影，不替代原始报告。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.evidence_root)
    report['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output/'table.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], rows=len(report['rows']), scans=report['scans'],
                         theory=report['theory']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
