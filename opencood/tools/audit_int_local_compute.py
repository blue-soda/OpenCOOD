"""Backfill executed U convolution MACs from verified stride-one crop areas.

This is standard dense-kernel arithmetic counting, not hardware instruction or
energy measurement. It does not count the rest of the detector as zero cost.
"""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from opencood.models.sub_modules.int_local_state import LocalStateDetector
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.launch_int_memory import digest
from opencood.tools.run_int_ego import write_json


def same_lattice_conv_macs(module, height, width):
    """MACs are proportional to input area ONLY for this checked U topology."""
    details = []
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            k = layer.kernel_size
            if (layer.stride != (1,1) or layer.dilation != (1,1) or
                    layer.padding != (k[0]//2,k[1]//2) or
                    k[0] % 2 != 1 or k[1] % 2 != 1 or layer.padding_mode != 'zeros'):
                raise ValueError('Cannot infer crop MACs from input area: '+name)
            macs_per_cell = layer.out_channels * (layer.in_channels//layer.groups) * k[0]*k[1]
            details.append(dict(layer=name, in_channels=layer.in_channels, out_channels=layer.out_channels,
                                kernel=list(k), groups=layer.groups, macs_per_cell=macs_per_cell))
    if not details:
        raise ValueError('No convolution found')
    return sum(d['macs_per_cell'] for d in details)*height*width, details


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--source-sha256',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reports',nargs='+',required=True)
    a=p.parse_args()
    for k,v in dict(mode='gru',stage='pilot',device='cpu',content_calibration=None,periodic_last=0,
                    max_age=.25,oracle_stage0=False,oracle_index=0,support_profile=False,zero_uncovered=False).items():
        setattr(a,k,v)
    teacher,cfg,_,provenance=setup(a)
    assert digest(a.checkpoint)==a.source_sha256
    saved=torch.load(a.checkpoint,map_location='cpu')
    model=LocalStateDetector(teacher,cfg,'recurrent').eval()
    model.load_state_dict(saved['model'],strict=True)
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    assert model.update.radius==4 and len(model.update.blocks)==2
    # This migrated U calls its projection once, then both BasicBlocks once.
    # BasicBlock's main and optional residual convs each execute once and retain
    # spatial size. Arbitrary/recurrent topology must not use this area shortcut.
    width,height,_=model.scatter.model_cfg['grid_size']
    width,height=int(width),int(height)
    height=(height+1)//2;width=(width+1)//2
    dense,details=same_lattice_conv_macs(model.update,height,width)
    assert len(details)==6 and (height,width)==(100,252)
    observed=[]
    hooks=[]
    for name,layer in model.update.named_modules():
        if isinstance(layer,nn.Conv2d):
            def record(module,inputs,output,name=name):
                observed.append(dict(layer=name,shape=list(output.shape)))
            hooks.append(layer.register_forward_hook(record))
    try:
        with torch.no_grad():
            model.update(torch.zeros(1,385,9,13))
    finally:
        for hook in hooks:hook.remove()
    assert sorted(r['layer'] for r in observed)==sorted(r['layer'] for r in details)
    assert all(r['shape'][0]==1 and r['shape'][-2:]==[9,13] for r in observed)
    rows=[]
    for filename in a.reports:
        report=json.loads(Path(filename).read_text())
        assert report['status']=='COMPLETED' and report['stage']=='full'
        assert report['checkpoint_sha256']==a.source_sha256 and report['weights_unchanged']
        assert report['scans']==5794 and report['labels']==1738
        for mode,result in report['results'].items():
            region=result['region_means']
            ratio=1. if region is None else region['executed_input_area_ratio']
            rows.append(dict(report=str(filename),report_sha256=digest(filename),mode=mode,
                scans=report['scans'],ap70=result['metrics']['bev_ap70'],
                mean_model_ms=result['mean_ms'],actual_write_fraction=1. if region is None else region['actual_write_fraction'],
                executed_U_input_area_ratio=ratio,mean_U_conv_gmacs=dense*ratio/1e9,
                mean_U_conv_gflops=2*dense*ratio/1e9,
                U_conv_compute_reduction_percent=100*(1-ratio),
                mean_U_executed_input_cells=height*width*ratio))
    write_json(Path(a.output),dict(status='COMPLETED',source_sha256=a.source_sha256,
        lattice=[height,width],dense_U_conv_macs=dense,dense_U_conv_flops=2*dense,
        counted_layers=details,small_crop_execution_trace=observed,rows=rows,
        convention='1 MAC = 2 FLOPs; standard Conv2d multiply/add counting including padded kernels.',
        method='Executed crop area includes batch copies, overlapping halos, reset and fallback; all six U convolutions retain H/W, so their summed standard MACs scale exactly with recorded executed area.',
        exclusions=['Not whole-model FLOPs: Pillar, E, heads, scorer, alignment, selection, copy/writeback omitted.',
                    'Bias, BN, activation and residual-add operations omitted; not treated as free in system evaluation.',
                    'Not measured GPU instructions, energy, or unique sparse dependency MACs.',
                    'Lower FLOPs/area is a positive compute signal even when current GPU latency worsens.',
                    'AP and execution costs must still be judged together; sparse/event hardware gains remain unverified.']))


if __name__=='__main__':
    main()
