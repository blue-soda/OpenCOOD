"""Execute indexed additions for ROM hits and ordinary MACs for misses on held-out detector tiles."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid
from opencood.tools.train_int_full_dair_pair import full_rows
from opencood.tools.check_int_silent_trajectory import scene_windows
from opencood.tools.audit_int_spike_pyramid_ops import ConvLedger
from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.models.sub_modules.int_planar_window import attach_window

def hybrid_tiles(conv,events,codes,scale,reference,first,items):
    assert conv.groups==1 and conv.padding_mode=='zeros'
    weight=conv.weight.cpu().numpy().transpose(2,3,1,0)
    kh,kw,_,outputs=weight.shape
    quantum=scale.cpu().numpy().reshape(1,1,-1,1)
    bank={k:(weight*quantum)*np.float32(k) for k in items}
    allocated=sum(v.nbytes for v in bank.values())
    assert allocated==4*len(items)*conv.weight.numel()
    code=codes[0].to(torch.int8).cpu().numpy()
    values=events[0].cpu().numpy()
    _,_,h,w=reference.shape
    side=min(4,h,w)
    result=[]
    for top,left in ((0,0),((h-side)//2,(w-side)//2)):
        yy,xx=np.indices((side,side)).reshape(2,-1)
        acc=np.zeros((side*side,outputs),dtype=np.float32)
        if first and conv.bias is not None: acc[:]=conv.bias.cpu().numpy()
        hits=misses=0
        for ky in range(kh):
            ys=(yy+top)*conv.stride[0]-conv.padding[0]+ky*conv.dilation[0]
            for kx in range(kw):
                xs=(xx+left)*conv.stride[1]-conv.padding[1]+kx*conv.dilation[1]
                valid,=np.nonzero((ys>=0)&(ys<code.shape[1])&(xs>=0)&(xs<code.shape[2]))
                levels=code[:,ys[valid],xs[valid]].T
                loc,ch=np.nonzero(levels)
                nonzero=levels[loc,ch].astype(np.int16)
                magnitude=abs(nonzero)
                hit=np.zeros(nonzero.size,dtype=bool)
                for k,scaled in bank.items():
                    for sign,operation in ((1,np.add),(-1,np.subtract)):
                        active=nonzero==sign*k
                        if active.any():
                            operation.at(acc,valid[loc[active]],scaled[ky,kx,ch[active]])
                            hit|=active
                where=valid[loc[~hit]]
                channels=ch[~hit]
                if channels.size:
                    amplitude=values[channels,ys[where],xs[where]]
                    np.add.at(acc,where,weight[ky,kx,channels]*amplitude[:,None])
                hits+=int(hit.sum())*outputs
                misses+=int((~hit).sum())*outputs
                assert int(hit.sum())==int(np.isin(magnitude,items).sum())
        expected=reference[0,:,top:top+side,left:left+side].cpu().numpy().reshape(outputs,-1).T
        peak=float(np.abs(expected).max())
        error=float(np.abs(expected-acc).max())
        assert error<=2e-4*max(peak,1.),(error,peak)
        result.append(dict(hits=hits,misses=misses,error=error,peak=peak,top=top,left=left))
    return result,allocated

@torch.no_grad()
def run(root,data,out,domain,kind):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    ck_path=root/('carry_'+kind)/'final.pt'
    train_tables=out/(kind+'_rom_train.json')
    learned=json.loads(train_tables.read_text())
    assert learned['parent']==fit.digest(ck_path) and learned['split']=='train'
    plan=learned['budgets']['16']
    manifest=root/('sequence_manifest.json' if domain=='dair' else 'formal_splits/v2v_official_deduplicated.jsonl')
    val=full_rows(manifest,data,'val') if domain=='dair' else v2v_rows(manifest,'val',data)
    rows=attach_window(scene_windows(val)[:5],False)
    assert [r['reset_before'] for r in rows]==[True,False,False,False,False]
    cfg=fit.load_yaml(str(root/'inputs/config.yaml'))
    cfg['data_augment']=[]
    src=fit.build_source(root/'inputs',cfg)
    net=RecoveryPyramid(src.core,root/'initial.pt','fine_temporal').cuda().eval()
    net.load_state_dict(torch.load(ck_path,map_location='cpu')['net'])
    if kind=='qat':
        for s in net.senders: s.continuous_threshold,s.payload_dtype=.5,'int8'
    ledger=ConvLedger(net)
    names={m:n for n,m in net.named_modules() if m in ledger.inputs}
    records=[]
    def checked(conv,events,first):
        ref=ledger.original(conv,events,first)
        if phase not in (0,4): return ref
        stage,sender=ledger.inputs[conv]
        scale=sender.step()*(sender.cap/127.) if kind=='qat' else sender.step()
        v=events/scale
        code=v.round()
        assert (v-code).abs().max()<1e-4
        tiles,size=hybrid_tiles(conv,events,code,scale,ref,first,plan['selected'][names[conv]][0 if first else 1])
        records.append(dict(phase=phase,layer=names[conv],tiles=tiles,bank_bytes=size,stage=stage))
        return ref
    state=None
    fit.delta_conv=checked
    try:
        for phase,(row,sample) in enumerate(zip(rows,fit.loader(rows,cfg,0,303)[1])):
            assert row['frame']==sample['meta']['frame']
            ego=fit.to_device(sample['ego'],torch.device('cuda'))
            _,state,_=net(fit.spatial_input(src,ego),src.core,state,ordinary=row['window_ordinary'])
    finally:
        fit.delta_conv=ledger.original
    assert len(records)==54 and sum(r['bank_bytes'] for r in records)==plan['extra_bytes']<=16*2**20
    with (out/(kind+'_tiles580.json')).open('x') as f:
        json.dump(dict(parent=learned['parent'],selection_sha=fit.digest(train_tables),rows=[(r['scene'],r['frame']) for r in rows],records=records),f)
if __name__=='__main__': run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5])
