"""Fit equal-capacity multiplier tables on training histograms and apply them unchanged to another split."""
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CAP=127
CENTER=1<<CAP
MASK=(1<<(2*CAP+1))-1

@functools.lru_cache(maxsize=8192)
def costs(table):
    """Shortest exact signed sum. Opposite-signed terms can be interleaved inside [-127,127]."""
    assert 1 in table and all(1<=c<=CAP for c in table)
    reached=frontier=CENTER
    depth=0
    result=np.zeros(CAP+1,dtype=np.int16)
    while reached!=MASK:
        extend=0
        for c in table:
            extend|=(frontier<<c)|(frontier>>c)
        frontier=extend & MASK & ~reached
        assert frontier
        reached|=frontier
        depth+=1
        pos=frontier>>CAP
        while pos:
            bit=pos & -pos
            result[bit.bit_length()-1]=depth
            pos-=bit
    return result

def greedy(hist):
    table=(1,)
    options={}
    old=int(hist @ costs(table))
    for size in range(1,33):
        if size in (4,8,16,32): options[size]=table
        if size==32: break
        best=None
        for c in range(2,CAP+1):
            if c in table: continue
            added=tuple(sorted((*table,c)))
            score=int(hist @ costs(added))
            if best is None or (score,c)<best[:2]: best=(score,c,added)
        if best[0]>=old:
            options.update({k:table for k in (4,8,16,32) if k>=size})
            break
        old,_,table=best
    return options

def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run(folder,kind,split):
    tp=folder/(kind+'_train.json')
    train=read(tp)
    assess=read(folder/(kind+'_'+split+'.json'))
    assert train['parent']==assess['parent'] and train['manifest']==assess['manifest']
    if split=='val': assert not set(map(tuple,train['rows'])).intersection(map(tuple,assess['rows']))
    out={'split':split,'parent':train['parent'],'training_input_sha':digest(tp),'evaluation_scans':assess['scans'],'budgets':{}}
    tables={}
    data={}
    locked=None
    if split=='val':
        locked=read(folder/(kind+'_tables_train.json'))
        assert locked['training_input_sha']==digest(tp) and locked['parent']==train['parent']
    for n,values in train['counts'].items():
        hist=np.array(values,dtype=np.int64)
        val=np.array(assess['counts'][n],dtype=np.int64)
        assert hist.shape==val.shape==(2,128)
        assert train['weight_elements'][n]==assess['weight_elements'][n]
        tables[n]=greedy(hist.sum(0)) if locked is None else {
            k:tuple(locked['budgets'][str(k)]['tables'][n]) for k in (4,8,16,32)}
        data[n]=val
    for k in (4,8,16,32):
        totals=np.zeros(2,dtype=np.int64)
        table_bytes=0
        for n,val in data.items():
            table=tables[n][k]
            totals+=val @ costs(table)
            table_bytes+=4*len(table)*assess['weight_elements'][n]
        out['budgets'][k]={'tables':{n:tables[n][k] for n in data},'add_g_per_scan':float(totals.sum()/assess['scans']/1e9),
            'phase_total_adds':totals.tolist(),'one_bank_fp32_bytes':table_bytes,'two_banks_fp32_bytes':2*table_bytes}
    native=np.sum(list(data.values()),axis=(0,1))
    nonzero=int(native[1:].sum())
    signed=[0]
    for i in range(1,128):
        j=i; n=0
        while j:
            if j%2: j-=2-j%4; n+=1
            j//=2
        signed.append(n)
    out['nonzero_g']=nonzero/assess['scans']/1e9
    out['standard_signed_g']=float(native @ signed /assess['scans']/1e9)
    out['full_direct_add_g']=out['nonzero_g']
    assert all(v['add_g_per_scan']>=out['nonzero_g']-1e-10 for v in out['budgets'].values())
    assert all(out['budgets'][a]['add_g_per_scan']>=out['budgets'][b]['add_g_per_scan']-1e-10 for a,b in zip((4,8,16),(8,16,32))) or split=='val'
    # The published frozen trace supplies an independent full-validation count.
    if split=='val':
        pub=read(folder.parent.parent/'qat_control_20261124'/assess['domain']/'verified549.json')['seeds']['707']['arms'][kind]
        assert pub['parent']==train['parent'] and abs(pub['signed_binary_adds']-out['standard_signed_g'])<1e-7
        assert abs(pub['nonzero_weighted_visits']-out['nonzero_g'])<1e-7
    with (folder/(kind+'_tables_'+split+'.json')).open('x') as f: json.dump(out,f,indent=2)
    print(kind,split,{k:round(v['add_g_per_scan'],3) for k,v in out['budgets'].items()})
if __name__=='__main__':
    run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
