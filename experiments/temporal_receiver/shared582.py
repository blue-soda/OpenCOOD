"""One phase-shared coefficient bank; include one resident folded dyadic base per weight."""
import functools
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def effective(raw, scales):
    assert set(scales) == set(raw['counts'])
    out = {}
    for name, (refresh, ordinary) in raw['counts'].items():
        multiplier = scales[name]
        assert multiplier in (1, 2)
        merged = {}
        for phase, codes in ((1, refresh), (multiplier, ordinary)):
            for k, n in enumerate(codes[1:], 1):
                if n:
                    amount = k * phase
                    merged[amount] = merged.get(amount, 0) + n
        assert sum(merged.values()) == sum(refresh[1:]) + sum(ordinary[1:])
        out[name] = merged
    return out


def choose(hist, sizes, budget):
    # A binary power base keeps every fallback amplitude exactly representable,
    # including for negative codes and ordinary-frame codes scaled by two.
    base = {n:max((1,2,4,8,16,32,64,128), key=lambda k:(h.get(k,0),-k))
            for n,h in hist.items()}
    banks = {n:sorted((k for k in h if k != base[n]), key=lambda k:(-h[k],k))
             for n,h in hist.items()}
    unit = functools.reduce(math.gcd,sizes.values())
    room = budget // unit
    best = np.zeros(room + 1,dtype=np.int64)
    track = {}
    for n in sorted(hist):
        step = sizes[n] // unit
        choice = np.zeros(room + 1,dtype=np.int16)
        cur = best.copy()
        gain = 0
        for j,k in enumerate(banks[n],1):
            offset = step * j
            if offset > room:
                break
            gain += hist[n][k]
            candidate = best[:-offset] + gain
            wins = candidate > cur[offset:]
            np.copyto(cur[offset:],candidate,where=wins)
            choice[offset:][wins] = j
        track[n] = choice
        best = cur
    selected = {}
    pos = room
    for n in reversed(sorted(hist)):
        j = int(track[n][pos])
        selected[n] = banks[n][:j]
        pos -= j * (sizes[n] // unit)
    used = sum(len(codes)*sizes[n] for n,codes in selected.items())
    assert used == (room-pos)*unit and used <= budget
    assert sum(hist[n][k] for n,codes in selected.items() for k in codes) == best[room]
    return base,selected,used


def toy_check():
    hist = {'a':{1:15,2:36,3:12,4:8}, 'b':{1:13,2:2,3:10,6:7}}
    sizes = {'a':12, 'b':20}
    for budget in range(0,111):
        base,chosen,_ = choose(hist,sizes,budget)
        items = [(n,k) for n,h in hist.items() for k in h if k != base[n]]
        actual = sum(hist[n][k] for n,codes in chosen.items() for k in codes)
        best = max(sum(hist[n][k] for n,k in subset)
                   for j in range(len(items)+1) for subset in itertools.combinations(items,j)
                   if sum(sizes[n] for n,k in subset) <= budget)
        assert actual == best


def run(folder):
    domain = folder.name
    published = read(folder.parent.parent / 'qat_control_20261124' / domain / 'verified549.json')['seeds']['707']['arms']
    result = {'domain':domain,'order':707,'models':{}}
    ids = {}
    for kind in ('native','qat'):
        train,val,probe = [read(folder / (kind+'_'+suffix+'.json')) for suffix in ('train','val','phase582')]
        assert train['parent'] == val['parent'] == probe['parent'] == published[kind]['parent']
        assert train['manifest'] == val['manifest'] and train['weight_elements'] == val['weight_elements']
        assert set(map(tuple,train['rows'])).isdisjoint(map(tuple,val['rows']))
        for split,raw in (('train',train),('val',val)):
            assert ids.setdefault(split,raw['rows']) == raw['rows']
        assert set(map(tuple,probe['rows'])).issubset(map(tuple,val['rows']))
        train_hist, val_hist = [effective(raw,probe['ordinary_multiplier']) for raw in (train,val)]
        factor = val['scans']*1e9
        visits = sum(sum(h.values()) for h in val_hist.values())
        assert abs(visits/factor - published[kind]['nonzero_weighted_visits']) < 1e-7
        sizes = {n:4*w for n,w in train['weight_elements'].items()}
        options = {}
        for mib in (0,4,16,64):
            base,extra,used = choose(train_hist,sizes,mib*2**20)
            resident = sum(h.get(base[n],0) for n,h in val_hist.items())
            extra_hits = sum(sum(h.get(k,0) for k in extra[n]) for n,h in val_hist.items())
            hits = resident + extra_hits
            options[mib] = dict(extra_bytes=used, hit_fraction=hits/visits, resident_hits_g=resident/factor,
                                extra_hits_g=extra_hits/factor, covered_g=published[kind]['covered_g']-hits/factor,
                                base=base, extra=extra)
        result['models'][kind] = dict(parent=train['parent'], scans=val['scans'], ap70=published[kind]['ap70'],
                                      max_scaled_error=probe['max_scaled_error'], budgets=options)
    with (folder/'shared582.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    for mib in (0,4,16,64):
        print(domain,mib,[round(m['budgets'][mib]['covered_g'],3) for m in result['models'].values()])


if __name__ == '__main__':
    toy_check()
    if len(sys.argv) > 1:
        run(Path(sys.argv[1]))
