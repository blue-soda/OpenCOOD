"""Validate independently saved wire/control predictions and serialized packet hashes."""
import json,hashlib
from pathlib import Path
import numpy as np
from infocom_wire import decode,ENVELOPE
r=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');wire=r/'infocom_paired_wire_smoke01';control=r/'infocom_paired_control_smoke01'
assert (wire/'summary.json').exists() and (control/'summary.json').exists()
index=json.loads((wire/'predictions_index.json').read_text());rows=[]
for row in index:
    a=np.load(str(wire/row['file']));b=np.load(str(control/row['file']))
    assert row['record_len']==[2]
    for key in ('pred','score','gt','frame_id'):assert np.array_equal(a[key],b[key]),key
    for message in row['wire']['messages']:
        packet=(wire/message['file']).read_bytes();assert hashlib.sha256(packet).hexdigest()==message['sha256']
        frame,sender,seq,*pose=ENVELOPE.unpack_from(packet)
        assert frame==int(row['frame_id']) and sender==1 and seq==0
        latent,mask=decode(packet[ENVELOPE.size:]);assert latent.shape==(256,) and mask.shape==(200,504)
    rows.append({'frame_id':row['frame_id'],'prediction_bitwise_equal':True,'wire_bytes':row['wire']['total_bytes']})
out={'passed':True,'frames':rows,'scope':'two-frame smoke only; exact latent/mask checked inside real forward; independent no-wire predictions equal'}
(r/'wire_smoke_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
