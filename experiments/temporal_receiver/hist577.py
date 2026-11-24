"""Collect magnitude counts weighted by actual valid convolution fan-out; leaves detector features unchanged."""
import json
import sys
from pathlib import Path
import torch
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid,periodic
from opencood.tools.train_int_full_dair_pair import full_rows
from opencood.tools.check_int_silent_trajectory import scene_windows
from opencood.tools.audit_int_spike_pyramid_ops import ConvLedger,signed_powers
from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.models.sub_modules.int_planar_window import attach_window

@torch.no_grad()
def run(root,data,out,domain,kind):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    cfg=fit.load_yaml(str(root/'inputs/config.yaml'))
    cfg['data_augment']=[]
    src=fit.build_source(root/'inputs',cfg)
    manifest=root/('sequence_manifest.json' if domain=='dair' else 'formal_splits/v2v_official_deduplicated.jsonl')
    path=root/('carry_'+kind)/'final.pt'
    ck=torch.load(path,map_location='cpu')
    assert ck['contract']['seed']==707 and ck['contract']['manifest']==fit.digest(manifest)
    net=RecoveryPyramid(src.core,root/'initial.pt','fine_temporal').cuda().eval()
    net.load_state_dict(ck['net'])
    if kind=='qat':
        assert ck['contract']['training_payload']=='int8'
        for s in net.senders: s.continuous_threshold,s.payload_dtype=.5,'int8'
    for split in ('train','val'):
        all_rows=full_rows(manifest,data,split) if domain=='dair' else v2v_rows(manifest,split,data)
        rows=attach_window(scene_windows(all_rows) if split=='train' else periodic(all_rows),False)
        ledger=ConvLedger(net,count_digits=True)
        convs={conv:name for name,conv in net.named_modules() if conv in ledger.inputs}
        weights={name:conv.weight.numel() for conv,name in convs.items()}
        hist={name:torch.zeros((2,128),device='cuda',dtype=torch.float64) for name in weights}
        def count(conv,events,first):
            answer=ledger.count(conv,events,first)
            _,sender=ledger.inputs[conv]
            q=sender.step()
            levels=events/(q*(sender.cap/127.)) if kind=='qat' else events/q
            codes=levels.round().long().abs()
            mask=codes.ne(0)
            taps=ledger.taps[(conv,*events.shape[-2:])].expand_as(codes)
            multiplicity=(taps[mask]*(conv.out_channels//conv.groups)).double()
            bins=torch.bincount(codes[mask],weights=multiplicity,minlength=128)
            hist[convs[conv]][0 if first else 1]+=bins
            return answer
        fit.delta_conv=count
        state=None
        try:
            for row,sample in zip(rows,fit.loader(rows,cfg,0,303)[1]):
                assert sample['meta']['frame']==row['frame']
                if row['reset_before']: state=None
                ego=fit.to_device(sample['ego'],torch.device('cuda'))
                _,state,_=net(fit.spatial_input(src,ego),src.core,state,ordinary=row['window_ordinary'])
        finally:
            fit.delta_conv=ledger.original
        all_bins=sum(hist.values())
        m=torch.arange(128,device='cuda',dtype=torch.float64)
        vectors=torch.stack((m.ne(0).double(),m,m.new_tensor([bin(i).count('1') for i in range(128)]),
                             m.new_tensor([len(signed_powers(i)) for i in range(128)])),dim=1)
        for phase,index in [('refresh',0),('ordinary',1)]:
            assert torch.equal(all_bins[index]@vectors,ledger.phase_digits[phase])
        result=dict(domain=domain,kind=kind,split=split,parent=fit.digest(path),manifest=fit.digest(manifest),
                    scans=len(rows),weight_elements=weights,rows=[[r['scene'],r['frame']] for r in rows],
                    counts={n:a.long().cpu().tolist() for n,a in hist.items()},
                    scope='Nonzero magnitude counts weighted by valid synaptic fan-out for each of 27 convolutions. Train uses first 5-scan window per eligible scene. Validation uses complete fixed-grid stream. Separate refresh and ordinary phases.')
        with (out/(kind+'_'+split+'.json')).open('x') as handle: json.dump(result,handle)
        print(domain,kind,split,len(rows),flush=True)

if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5])
