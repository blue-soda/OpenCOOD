"""Measure GPU board energy for frozen residual-mask projection replays."""
import argparse
import ctypes
import json
from pathlib import Path
from statistics import mean
import subprocess
import time

import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_phase_projection import DeepPhaseProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import TILE
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_deep_block_cache import fingerprint
from opencood.tools.time_int_deep_phase_transfer import DEVICE, MODES, audit, clips, inputs
from opencood.tools.time_int_periodic_compact import exclusive


ORDER = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
MEASURED_PASSES = 3
IDLE_S = 4


class BoardEnergy:
    """Read the driver's cumulative board counter in millijoules."""

    def __init__(self, uuid):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlInit_v2())
        self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle)))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.nvmlErrorString(code).decode())

    def mark(self):
        value = ctypes.c_ulonglong()
        started = time.perf_counter_ns()
        self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)))
        ended = time.perf_counter_ns()
        return dict(counter_mj=value.value, time_ns=(started+ended)//2, read_ns=ended-started)

    def telemetry(self):
        def query(function, *args):
            value = ctypes.c_uint()
            self.check(function(self.handle,*args,ctypes.byref(value)))
            return value.value

        return dict(temperature_c=query(self.lib.nvmlDeviceGetTemperature,0),
                    sm_clock_mhz=query(self.lib.nvmlDeviceGetClockInfo,1),
                    power_mw=query(self.lib.nvmlDeviceGetPowerUsage))


def interval(start, end):
    delta = end['counter_mj']-start['counter_mj']
    seconds = (end['time_ns']-start['time_ns'])/1e9
    assert delta>0 and seconds>0
    return delta/1000., seconds


def stage(plan, rows, cfg, traces, uuid):
    entries = []
    for clip, data in clips(plan,rows,cfg,traces,uuid):
        sequence = []
        for i,sample,ref,masks in data:
            ego = dict(processed_lidar=to_device(sample['ego']['processed_lidar'],DEVICE))
            optional = None if masks is None else masks['optional']
            sequence.append((i, sample['meta'], ego, optional, ref['refresh']))
        entries.append((clip,sequence))
    torch.cuda.synchronize()
    return entries


def replay(name, source, model, entries, proofs=None):
    for clip,sequence in entries:
        state = None
        for i,meta,ego,optional,expected_refresh in sequence:
            x = spatial_input(source,ego)
            pred,state,info = model.step_spatial(x,state,meta,requests=None if name=='p5' else optional)
            torch.cuda.synchronize()
            assert info['full_refresh']==expected_refresh
            if proofs is not None and i%16==0:
                expected = proofs[clip,meta['frame']]['arms'][name]['fingerprint']
                assert expected is not None and fingerprint(pred,state)==expected


def idle(meter, uuid):
    exclusive(uuid)
    torch.cuda.synchronize()
    start = meter.mark()
    time.sleep(IDLE_S)
    end = meter.mark()
    joules,seconds = interval(start,end)
    exclusive(uuid)
    return dict(start=start,end=end,seconds=seconds,joules=joules,watts=joules/seconds)


def run_blocks(meter, args, out, source, models, entries, proofs):
    scans_per_pass = sum(len(sequence) for _,sequence in entries)
    scans = MEASURED_PASSES*scans_per_pass
    arms, baselines = [], []
    with (out/'board_intervals.jsonl').open('x',buffering=1) as raw:
        def record(row):
            raw.write(json.dumps(row,allow_nan=False)+'\n')

        for block,order in enumerate(ORDER):
            baseline = dict(kind='idle',block=block,**idle(meter,args.uuid))
            baselines.append(baseline)
            record(baseline)
            for slot,index in enumerate(order):
                name = MODES[index]
                exclusive(args.uuid)
                replay(name,source,models[name],entries,proofs)
                before = meter.telemetry()
                start = meter.mark()
                for _ in range(MEASURED_PASSES):
                    replay(name,source,models[name],entries)
                end = meter.mark()
                after = meter.telemetry()
                exclusive(args.uuid)
                joules,seconds = interval(start,end)
                row = dict(kind='workload',block=block,slot=slot,arm=name,start=start,end=end,
                           before=before,after=after,scans=scans,joules=joules,seconds=seconds,
                           gross_joules_per_scan=joules/scans,
                           above_idle_joules_per_scan=(joules-baseline['watts']*seconds)/scans,
                           elapsed_ms_per_scan=1000*seconds/scans,mean_board_watts=joules/seconds)
                record(row)
                arms.append(row)
                write_json(out/'status.json',dict(status='RUNNING',phase='board_intervals',
                                                  completed=len(arms),total=len(ORDER)*len(MODES)))
        terminal = dict(kind='idle',block=len(ORDER),**idle(meter,args.uuid))
        baselines.append(terminal)
        record(terminal)
    return arms,baselines


@torch.no_grad()
def run(args,out):
    exclusive(args.uuid)
    meter = BoardEnergy(args.uuid)
    meter.mark()
    paths,hashes,plan,traces,costs = inputs(args)
    args.kind = 'geometry'
    source,cfg,rows,_ = build(args)
    source.period,source.max_age = 5,.55
    weights = model_digest(source)
    models = {'p5':source}
    for name in MODES[1:]:
        projector = None if name=='crop' else DeepPhaseProjection(source.core.deblocks[2], indexed=name=='phase_index')
        models[name] = DeepBlockCache(source,TILE,'light',planner=plan_host_coalesced_refresh,crop_first=True,
                                      deep_projector=projector)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        device=torch.cuda.get_device_name(),uuid=args.uuid,torch=torch.__version__,source_sha256=SOURCE_SHA,
        driver=subprocess.check_output(['nvidia-smi','-i',args.uuid,'--query-gpu=driver_version','--format=csv,noheader']).decode().strip(),
        inputs_sha256=hashes,arms=MODES,order=[[MODES[i] for i in block] for block in ORDER],
        scenes=[s[0] for s in plan['scenes']],scans_per_pass=sum(map(len,plan['clips'])),
        measured_passes_per_interval=MEASURED_PASSES,warm_passes_per_interval=1,idle_s=IDLE_S,
        primary='Gross GPU board joules per scan from synchronized nvmlDeviceGetTotalEnergyConsumption endpoints.',
        boundary='Pillar to heads on GPU-resident inputs and fixed optional masks; one host/device barrier per scan. Board only. Replays contain refresh and scene resets. Excludes input loading, H2D, online mask scoring and postprocessing. Idle correction is diagnostic.'))
    proofs = audit(source,models,cfg,rows,plan,traces,costs,args.uuid,out)
    assert len(proofs)==272
    entries = stage(plan,rows,cfg,traces,args.uuid)
    assert sum(len(s) for _,s in entries)==len(proofs)
    arms,baselines = run_blocks(meter,args,out,source,models,entries,proofs)
    assert model_digest(source)==weights and all(digest(paths[k])==h for k,h in hashes.items())
    stats = {}
    for arm in MODES:
        data = [r for r in arms if r['arm']==arm]
        stats[arm] = dict(scans=sum(r['scans'] for r in data),intervals=len(data),ap50=None,ap70=None,
            parameters=sum(p.numel() for p in source.parameters()),
            core_conv_flops_per_scan=mean(r['arms'][arm]['core_conv_flops'] for r in proofs.values()),
            **{key:mean(r[key] for r in data) for key in ('gross_joules_per_scan','above_idle_joules_per_scan',
                                                        'elapsed_ms_per_scan','mean_board_watts')})
    paired = {}
    for control in MODES[:-1]:
        paired[control] = []
        for block in range(len(ORDER)):
            by_arm = {r['arm']:r for r in arms if r['block']==block}
            paired[control].append(by_arm[control]['gross_joules_per_scan']-by_arm['phase_index']['gross_joules_per_scan'])
    write_json(out/'report.json',dict(status='COMPLETED',metric='gpu_board_energy_counter',
        verified_scans=len(proofs),workload_intervals=len(arms),metrics=stats,paired_gross_saving_j_per_scan=paired,
        idle_board_watts=[r['watts'] for r in baselines],source_weights_unchanged=True,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    write_json(out/'status.json',dict(status='COMPLETED',scans_each=stats['p5']['scans'],intervals=len(arms)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','plan','final-run','uuid','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
