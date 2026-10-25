"""Independent CPU verification of serialized packets and no-wire prediction parity."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from infocom_wire import decode,ENVELOPE
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');wire=r/'infocom_paired_wire_full01';control=r/'infocom_allpairs_full01_compat'
deadline=time.time()+14400
while not all((p/'summary.json').exists() for p in (wire,control)):
    if any((p/'failure.json').exists() for p in (wire,control)) or time.time()>deadline:
        raise RuntimeError('Incomplete/failed prerequisite; no retry')
    time.sleep(30)
index=json.loads((wire/'predictions_index.json').read_text());control_index=json.loads((control/'predictions_index.json').read_text())
ids=json.loads((r/'common_val.json').read_text());assert len(index)==len(control_index)==len(ids)==1738
rows=[];all_exact=True;all_close=True
for row,reference,fid in zip(index,control_index,ids):
    assert row['frame_id']==reference['frame_id']==fid and row['record_len']==reference['record_len']==[2]
    a_path=wire/row['file'];b_path=control/reference['file']
    for p,entry in ((a_path,row),(b_path,reference)):assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
    a=np.load(str(a_path));b=np.load(str(b_path));exact=True;close=True;error={}
    for key in ('pred','score','gt'):
        same=a[key].shape==b[key].shape
        exact=exact and same and np.array_equal(a[key],b[key])
        close=close and same and np.allclose(a[key],b[key],rtol=0,atol=1e-5)
        error[key]=float(np.max(np.abs(a[key]-b[key]))) if same and a[key].size else (0. if same else None)
    nbytes=0
    for message in row['wire']['messages']:
        packet=(wire/message['file']).read_bytes();assert hashlib.sha256(packet).hexdigest()==message['sha256']
        frame,sender,seq,*pose=ENVELOPE.unpack_from(packet)
        assert frame==int(fid) and sender==1 and seq==0 and np.isfinite(pose).all()
        latent,mask=decode(packet[ENVELOPE.size:]);assert latent.shape==(256,) and mask.shape==(200,504)
        assert np.isfinite(latent).all() and np.isfinite(mask).all()
        assert len(packet)==message['total_bytes'];nbytes+=len(packet)
    assert len(row['wire']['messages'])==1 and nbytes==row['wire']['total_bytes']
    all_exact=all_exact and exact;all_close=all_close and close
    rows.append({'frame_id':fid,'bytes':nbytes,'prediction_exact':exact,'prediction_close_1e5':close,'max_abs_error':error})
values=np.array([x['bytes'] for x in rows]);summary=json.loads((wire/'summary.json').read_text())
assert abs(values.mean()-summary['wire_statistics']['total_bytes']['mean'])<1e-9
report={'samples':len(rows),'packet_sha_and_decode_verified':len(rows),'all_predictions_bitwise_equal':bool(all_exact),'all_predictions_close_1e5':bool(all_close),'bytes_mean':float(values.mean()),'bytes_p95':float(np.percentile(values,95)),'bytes_max':int(values.max()),'rows':rows}
(r/'wire_full_cpu_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
