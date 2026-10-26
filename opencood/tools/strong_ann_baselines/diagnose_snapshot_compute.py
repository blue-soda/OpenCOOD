"""Inspect immutable-reference discrepancies without relaxing the timing gate."""
import json
from pathlib import Path
import numpy as np
import torch
from opencood.tools.strong_ann_baselines.compute_harness import fingerprint


def diagnose(adapter,indices,references,reference,output):
    rows=[];previous={};manifest=json.loads((reference/'predictions/index.json').read_text())['frames']
    with torch.no_grad():
        for repeat in range(2):
            for index in indices:
                fid,batch=adapter.prepare_cpu(index);adapter.reset(fid)
                result=adapter.infer(adapter.to_device(batch));arrays=adapter.to_cpu(result)
                packets=adapter.packets();actual=fingerprint(fid,arrays,packets);expected=references[fid]
                changed=[k for k in expected if actual.get(k)!=expected[k]]
                row=dict(index=index,frame_id=fid,repeat=repeat,changed_fields=changed,
                         packets_exact=actual['packets']==expected['packets'],
                         equal_previous=None if fid not in previous else actual==previous[fid])
                if changed:
                    prefix=output/('mismatch_r%d_i%04d'%(repeat,index));prefix.mkdir()
                    np.savez_compressed(str(prefix/'actual.npz'),**arrays)
                    (prefix/'fingerprints.json').write_text(json.dumps(dict(actual=actual,expected=expected),indent=2))
                    for i,(label,blob) in enumerate(packets):(prefix/('packet%d.bin'%i)).write_bytes(blob)
                    with np.load(str(reference/'predictions'/manifest[index]['file'])) as z:
                        for key,refkey in [('boxes','pred_boxes'),('scores','scores')]:
                            a,r=arrays[key],z[refkey];row[key+'_shape_equal']=a.shape==r.shape
                            if a.shape==r.shape:
                                diff=np.abs(a.astype(np.float64)-r.astype(np.float64))
                                row[key+'_max_abs']=float(diff.max()) if diff.size else 0.
                                row[key+'_changed_values']=int(np.count_nonzero(a!=r))
                previous[fid]=actual;rows.append(row)
                tmp=output/'diagnostic_progress.tmp';tmp.write_text(json.dumps(rows,indent=2));tmp.replace(output/'diagnostic_progress.json')
    summary=dict(diagnostic_only=True,no_timing_or_AP_claim=True,reference_gate_not_relaxed=True,
                 unique_frames=len(indices),calls=len(rows),mismatched_calls=sum(bool(r['changed_fields']) for r in rows),
                 packet_mismatched_calls=sum(not r['packets_exact'] for r in rows),
                 unstable_repeat_calls=sum(r['equal_previous'] is False for r in rows),rows=rows)
    (output/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    return summary
