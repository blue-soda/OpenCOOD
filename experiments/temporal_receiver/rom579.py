"""Pick per-phase direct-lookup hits under a single detector-wide extra ROM budget, using training only."""
import json
import math
import sys
import functools
from pathlib import Path
import numpy as np

def read(p): return json.loads(p.read_text())
def select(train,budget):
    # One extra FP32 scaled coefficient per original weight for each (layer,phase,magnitude).
    slots=[(name,phase) for name in sorted(train['counts']) for phase in range(2)]
    sizes=[4*train['weight_elements'][name] for name,phase in slots]
    unit=functools.reduce(math.gcd,sizes)
    room=budget//unit
    best=np.zeros(room+1,dtype=np.int64)
    choices=[]
    values=[]
    for (name,phase),size in zip(slots,sizes):
        freq=train['counts'][name][phase]
        order=sorted((k for k in range(1,128) if freq[k]),key=lambda k:(-freq[k],k))
        values.append(order)
        step=size//unit
        take=np.zeros(room+1,dtype=np.uint8)
        cur=best.copy()
        gain=0
        for n,k in enumerate(order,1):
            offset=n*step
            if offset>room: break
            gain+=freq[k]
            score=best[:-offset]+gain
            change=score>cur[offset:]
            np.copyto(cur[offset:],score,where=change)
            take[offset:][change]=n
        choices.append(take)
        best=cur
    pos=room
    assigned={}
    for i in range(len(slots)-1,-1,-1):
        n=int(choices[i][pos])
        name,phase=slots[i]
        assigned.setdefault(name,[[],[]])[phase]=values[i][:n]
        pos-=n*sizes[i]//unit
    spent=(room-pos)*unit
    return assigned,spent,int(best[room])

def hits(data,selected):
    return sum(sum(data['counts'][name][phase][k] for k in codes)
               for name,phases in selected.items() for phase,codes in enumerate(phases))

def run(folder,kind,split):
    train=read(folder/(kind+'_train.json'))
    assess=read(folder/(kind+'_'+split+'.json'))
    assert assess['parent']==train['parent'] and assess['weight_elements']==train['weight_elements']
    locked=read(folder/(kind+'_rom_train.json')) if split=='val' else None
    if locked: assert locked['parent']==train['parent']
    totals=sum(sum(sum(row[1:]) for row in pair) for pair in assess['counts'].values())
    result={'parent':train['parent'],'split':split,'scans':assess['scans'],'nonzero_g':totals/assess['scans']/1e9,'budgets':{}}
    for mib in (0,4,16,64):
        if locked is None:
            selected,used,gain=select(train,mib*2**20)
            assert gain==hits(train,selected)
        else:
            prior=locked['budgets'][str(mib)]
            selected,used=prior['selected'],prior['extra_bytes']
        assert used==sum(len(codes)*4*train['weight_elements'][n] for n,phases in selected.items() for codes in phases)
        assert used<=mib*2**20
        saved=hits(assess,selected)
        item=dict(extra_bytes=used,lookup_fraction=saved/totals,saved_g=saved/assess['scans']/1e9,
                  residual_hybrid_g=(2*totals-saved)/assess['scans']/1e9,selected=selected)
        if split=='val':
            verified=read(folder/(kind+'_tables_val.json'))
            assert abs(verified['nonzero_g']-result['nonzero_g'])<1e-8
            archive=read(folder.parent.parent/'qat_control_20261124'/assess['domain']/'verified549.json')['seeds']['707']['arms'][kind]
            item['covered_hybrid_g']=archive['covered_g']-item['saved_g']
        result['budgets'][mib]=item
    with (folder/(kind+'_rom_'+split+'.json')).open('x') as f:
        json.dump(result,f,allow_nan=False)
    print(kind,split,[(b,round(v['lookup_fraction']*100,2),round(v['saved_g'],3)) for b,v in result['budgets'].items()])
if __name__=='__main__':
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
