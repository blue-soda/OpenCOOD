"""Bounded CPU-only whole-split lossless recoding of saved actual messages.

Preserves 80-byte envelopes and proves identical decoder inputs. This is not
a fresh GPU forward/AP measurement or a latency benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from opencood.tools.strong_ann_baselines import infocom_wire as old
from opencood.tools.strong_ann_baselines import infocom_wire_q4_exact as new


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source,output):
    output.mkdir(parents=True,exist_ok=False);(output/'wire').mkdir()
    index=json.loads((source/'predictions_index.json').read_text())
    ids=json.loads((source.parent/'common_val.json').read_text())
    assert len(index)==len(ids)==1738
    proof=json.loads((source.parent/'wire_full_cpu_audit.json').read_text())
    assert proof['samples']==1738 and proof['all_predictions_bitwise_equal']
    started=time.monotonic();rows=[];sizes=[];originals=[];canonical_errors=[]
    for entry,fid in zip(index,ids):
        assert entry['frame_id']==fid and entry['record_len']==[2]
        assert len(entry['wire']['messages'])==1
        message=entry['wire']['messages'][0]
        path=source/message['file'];original=path.read_bytes();assert digest(path)==message['sha256']
        envelope=original[:old.ENVELOPE.size]
        frame,sender,seq,*pose=old.ENVELOPE.unpack(envelope)
        assert frame==int(fid) and sender==1 and seq==0 and np.isfinite(pose).all()
        latent,mask=old.decode(original[old.ENVELOPE.size:])
        packet,record=new.encode(latent,mask);decoded_l,decoded_m=new.decode(packet)
        assert decoded_l.tobytes()==latent.tobytes() and decoded_m.tobytes()==mask.tobytes()
        paid=envelope+packet;assert len(paid)<=len(original)
        target=output/'wire'/path.name;target.write_bytes(paid)
        # Independent disk reread, not only the encoder's in-memory buffers.
        reread=target.read_bytes();a,b=new.decode(reread[old.ENVELOPE.size:])
        assert reread[:old.ENVELOPE.size]==envelope and a.tobytes()==latent.tobytes() and b.tobytes()==mask.tobytes()
        canonical=np.rint(np.clip(mask,0,1)*np.float32(15))/np.float32(15)
        error=float(np.max(np.abs(canonical-mask)));canonical_errors.append(error)
        record.update(frame_id=fid,file='wire/'+path.name,sha256=digest(target),
            envelope_bytes=len(envelope),total_bytes=len(paid),source_total_bytes=len(original),
            source_sha256=message['sha256'],canonical_max_abs_error=error)
        rows.append(record);sizes.append(len(paid));originals.append(len(original))
    (output/'index.json').write_text(json.dumps(rows,indent=2))
    summary=dict(passed=True,samples=1738,all_decoder_inputs_bitwise_equal=True,
        all_envelopes_identical=True,source_index_sha256=digest(source/'predictions_index.json'),
        source_wire_audit_sha256=digest(source.parent/'wire_full_cpu_audit.json'),
        script_sha256=digest(Path(__file__)),codec_sha256=digest(Path(new.__file__)),
        mean_bytes=float(np.mean(sizes)),p95_bytes=float(np.percentile(sizes,95)),max_bytes=max(sizes),
        original_mean_bytes=float(np.mean(originals)),total_bytes=sum(sizes),
        reduction_fraction=1-sum(sizes)/sum(originals),
        layouts={k:sum(row['format']==k for row in rows) for k in ('ICW1','ICQ1')},
        canonical_max_abs_error=max(canonical_errors),
        source_common_gt_ap=json.loads((source/'common_gt_replay.json').read_text())['common_gt_bev_ap'],
        status='CPU lossless recoding verified; AP inherited via identical inputs, no new GPU AP or latency run',
        seconds=time.monotonic()-started,index_sha256=digest(output/'index.json'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
