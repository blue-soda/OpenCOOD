"""Inspect immutable-reference discrepancies without relaxing the timing gate."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
try:
    from .compute_harness import fingerprint
except ImportError:
    from compute_harness import fingerprint


def diagnose(adapter,indices,references,reference,output):
    rows=[];previous={}
    snn=(reference/'predictions/index.json').exists()
    manifest=(json.loads((reference/'predictions/index.json').read_text())['frames'] if snn
              else json.loads((reference/'predictions_index.json').read_text()))
    # The only discrepant SNN frame identified in the preceding 64-frame audit.
    # Observe the integer encoder output and continuous sender drive; never
    # replace either tensor. This instrumentation is strictly untimed.
    trace={};handle=None
    model=getattr(adapter,'model',None)
    if snn and model is not None:
        def sender_hook(module,inputs,result):
            if adapter.fid=='004276':
                trace['encoder_counts']=inputs[0].detach().cpu().numpy().copy()
                trace['projected_drive']=result.detach().cpu().numpy().copy()
        handle=model.content.project.register_forward_hook(sender_hook)
    with torch.no_grad():
        for repeat in range(2):
            for index in indices:
                fid,batch=adapter.prepare_cpu(index);adapter.reset(fid);trace.clear()
                result=adapter.infer(adapter.to_device(batch));arrays=adapter.to_cpu(result)
                packets=adapter.packets();actual=fingerprint(fid,arrays,packets);expected=references[fid]
                changed=[k for k in expected if actual.get(k)!=expected[k]]
                row=dict(index=index,frame_id=fid,repeat=repeat,changed_fields=changed,
                         packets_exact=actual['packets']==expected['packets'],
                         equal_previous=None if fid not in previous else actual==previous[fid])
                if trace:
                    row['sender_trace']={k:dict(shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in trace.items()}
                    if repeat==0:np.savez_compressed(str(output/('sender_trace_i%04d.npz'%index)),**trace)
                if changed:
                    prefix=output/('mismatch_r%d_i%04d'%(repeat,index));prefix.mkdir()
                    np.savez_compressed(str(prefix/'actual.npz'),**arrays)
                    (prefix/'fingerprints.json').write_text(json.dumps(dict(actual=actual,expected=expected),indent=2))
                    for i,(label,blob) in enumerate(packets):(prefix/('packet%d.bin'%i)).write_bytes(blob)
                    refpath=(reference/'predictions'/manifest[index]['file'] if snn else reference/manifest[index]['file'])
                    with np.load(str(refpath)) as z:
                        for key,refkey in [('boxes','pred_boxes' if snn else 'pred'),('scores','scores' if snn else 'score')]:
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
    if handle is not None:handle.remove()
    return summary
