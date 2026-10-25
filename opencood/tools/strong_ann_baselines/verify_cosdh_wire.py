"""Independent wire-packet and prediction verification against official CoSDH."""
import argparse,json,hashlib,struct,time
from pathlib import Path
import numpy as np
from cosdh_wire import HEADER,unpack,decode_request,decode_latent,decode_boxes
from infocom_wire import ENVELOPE
p=argparse.ArgumentParser();p.add_argument('--run',default='cosdh_wire_full01');p.add_argument('--smoke',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
root=Path('/data1/zhh/gzc/diagnostics/ann_baselines_20261025');run=root/a.run;control=root/'cosdh_allpairs_full01';deadline=time.time()+14400
while not (run/'summary.json').exists():
    if not a.wait or (run/'failure.json').exists() or time.time()>deadline:raise RuntimeError('Incomplete run')
    time.sleep(30)
rows=[json.loads(x) for x in (run/'frames.jsonl').read_text().splitlines()]
references=[json.loads(x) for x in (control/'frames.jsonl').read_text().splitlines()]
ids=json.loads((root/'common_val.json').read_text());records=[];all_exact=True
if not a.smoke:assert len(rows)==len(ids)==1738
for row,ref,fid in zip(rows,references,ids):
    assert row['frame_id']==ref['frame_id']==fid and row['record_len']==[2]
    for base,item in ((run,row),(control,ref)):
        assert hashlib.sha256((base/item['file']).read_bytes()).hexdigest()==item['sha256']
    received=np.load(str(run/row['file']));original=np.load(str(control/ref['file']))
    exact=all(np.array_equal(received[k],original[k]) for k in ('pred','score','gt','frame_id'));all_exact &= exact
    counts={};late_count=None;latent_shapes=[]
    for item in row['wire']['packets']:
        packet=(run/item['file']).read_bytes();assert len(packet)==item['bytes'] and hashlib.sha256(packet).hexdigest()==item['sha256']
        frame,sender,sequence,*pose=ENVELOPE.unpack_from(packet);assert frame==int(fid) and sequence==0 and np.isfinite(pose).all()
        if item['direction']=='demand':
            assert sender==0
            mask=decode_request(packet[ENVELOPE.size:]);assert mask.shape==(200,504)
        else:
            assert sender==1
            magic,n=struct.unpack_from('<4sI',packet,ENVELOPE.size);assert magic==b'CFW1' and n==4
            offset=ENVELOPE.size+8
            for stage in range(4):
                fields=HEADER.unpack_from(packet,offset);length=HEADER.size+fields[8];part=packet[offset:offset+length];offset+=length
                if stage<3:
                    number,latent,target=decode_latent(part);assert number==stage and np.isfinite(latent).all() and np.isfinite(target).all();latent_shapes.append(list(latent.shape))
                else:
                    boxes,late_pose=decode_boxes(part);assert np.isfinite(boxes).all() and np.isfinite(late_pose).all();late_count=len(boxes)
            assert offset==len(packet)
        counts[item['direction']]=len(packet)
    assert counts['demand']+counts['response']==row['wire']['total_bytes'] and late_count==row['wire']['late_boxes']
    records.append({'frame_id':fid,'prediction_bitwise_equal':bool(exact),'total_bytes':row['wire']['total_bytes'],'late_boxes':late_count,'latent_shapes':latent_shapes})
values=np.array([r['total_bytes'] for r in records])
report={'samples':len(records),'full_split':len(records)==1738,'all_predictions_bitwise_equal':bool(all_exact),'packet_sha_and_decode_verified':len(records)*2,'bytes_mean':float(values.mean()),'bytes_p95':float(np.percentile(values,95)),'bytes_max':int(values.max()),'rows':records}
(run/'wire_cpu_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if not all_exact:raise RuntimeError('Wire path changed predictions')
