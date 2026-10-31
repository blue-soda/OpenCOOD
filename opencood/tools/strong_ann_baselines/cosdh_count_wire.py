"""Reference sparse integer count transport; all stages, masks and request paid.

Counts are selected on GPU before transport. Each payload is encoded/decoded
once during explicit wire evaluation only. No FP16 feature payload or late box.
"""
import hashlib
import struct
from pathlib import Path
import numpy as np
from cosdh_wire import pack, unpack, encode_request, decode_request
from infocom_wire import envelope


def encode_counts(stage, counts, scale, steps):
    x = np.asarray(counts)
    assert x.ndim == 3 and np.isfinite(x).all() and np.all(x == np.floor(x))
    assert x.min() >= 0 and x.max() <= steps and steps in (8, 32)
    c,h,w = x.shape; scale = np.asarray(scale, dtype='<f4').reshape(-1)
    assert len(scale) == c and np.isfinite(scale).all() and (scale > 0).all()
    indices = np.flatnonzero(x.reshape(-1)).astype('<u4')
    values = x.reshape(-1)[indices].astype('u1')
    return pack(3, stage, h, w, c, len(indices), scale.tobytes()+indices.tobytes()+values.tobytes(), steps)


def decode_counts(packet):
    kind,stage,steps,h,w,c,n,raw = unpack(packet)
    assert kind == 3 and steps in (8, 32) and len(raw) == c*4+n*5
    scale = np.frombuffer(raw[:c*4], dtype='<f4').copy()
    indices = np.frombuffer(raw[c*4:c*4+n*4], dtype='<u4')
    values = np.frombuffer(raw[c*4+n*4:], dtype='u1')
    assert np.isfinite(scale).all() and (scale > 0).all()
    assert len(indices)==0 or (indices[-1] < c*h*w and np.all(indices[1:] > indices[:-1]))
    assert np.all((values > 0) & (values <= steps))
    counts = np.zeros(c*h*w, dtype='u1'); counts[indices] = values
    return stage,steps,counts.reshape(c,h,w),scale


def encode_mask(stage, mask):
    x = np.asarray(mask, dtype='<f4'); assert x.ndim == 2 and np.isfinite(x).all()
    binary = np.all((x==0)|(x==1)); h,w = x.shape
    raw = np.packbits(x.astype('u1').reshape(-1)).tobytes() if binary else x.tobytes()
    return pack(4, stage, h, w, 1, int(np.count_nonzero(x)), raw, 0 if binary else 1)


def decode_mask(packet):
    kind,stage,flag,h,w,c,n,raw = unpack(packet)
    assert kind==4 and c==1 and flag in (0,1)
    if flag == 0:
        assert len(raw)==(h*w+7)//8
        x=np.unpackbits(np.frombuffer(raw,dtype='u1'))[:h*w].astype('f4').reshape(h,w)
    else:
        assert len(raw)==h*w*4
        x=np.frombuffer(raw,dtype='<f4').copy().reshape(h,w)
    assert np.isfinite(x).all() and np.count_nonzero(x)==n
    return stage,x


class CoSDHCountWire:
    def __init__(self, model, counts, output):
        self.counts=counts; self.handles=[]; self.output=Path(output)/'wire';self.output.mkdir()
        for stage in range(3):
            self.handles += [model.naive_compressor_list[stage].decoder.register_forward_pre_hook(self.count_hook(stage)),
                model.fusion_net[stage].naive_communication.register_forward_pre_hook(self.request_hook),
                model.fusion_net[stage].naive_communication.register_forward_hook(self.mask_hook(stage))]

    def begin(self, batch, frame_id):
        assert batch['ego']['record_len'].tolist()==[2]
        self.frame_id=frame_id;self.pose=batch['ego']['pairwise_t_matrix'][0,0,1].detach().cpu().numpy()
        self.request=None;self.request_array=None;self.parts={};self.details={}

    def count_hook(self, stage):
        def apply(module, inputs):
            import torch
            count=self.counts.current[stage]; scale=self.counts.scales[stage]
            packet=encode_counts(stage,count.cpu().numpy(),scale.cpu().numpy(),self.counts.steps)
            got_stage,steps,got,got_scale=decode_counts(packet)
            assert got_stage==stage and steps==self.counts.steps
            restored=torch.from_numpy(got.astype('f4')).to(count.device)*(torch.from_numpy(got_scale).to(count.device).reshape(-1,1,1)/steps)
            assert torch.equal(restored,inputs[0][1]), 'Count wire changed decoder input'
            replacement=inputs[0].clone();replacement[1]=restored
            assert stage not in self.parts
            self.parts[stage]=packet
            self.details[stage]=dict(stage=stage,count_section_bytes=len(packet),selected_events=int(count.sum()),
                nonzero_counts=int(torch.count_nonzero(count)),decoder_input_exact=True,scale_metadata_included=True)
            return (replacement,)+inputs[1:]
        return apply

    def request_hook(self, module, inputs):
        import torch
        req=inputs[3];assert req is not None
        x=req[0,0].detach().cpu().numpy()
        if self.request is None:
            self.request=encode_request(x);self.request_array=decode_request(self.request)
        assert np.array_equal(x,self.request_array)
        got=req.clone();got[0,0]=torch.from_numpy(self.request_array).to(req.device)
        return inputs[:3]+(got,)

    def mask_hook(self, stage):
        def apply(module, inputs, output):
            import torch
            masks=output[0];assert masks.shape[0]==2
            x=masks[1,0].detach().cpu().numpy();packet=encode_mask(stage,x);got_stage,got=decode_mask(packet)
            assert got_stage==stage and np.array_equal(x,got)
            replacement=masks.clone();replacement[1,0]=torch.from_numpy(got).to(masks.device)
            self.parts[stage+3]=packet;self.details[stage]['mask_section_bytes']=len(packet)
            return (replacement,)+output[1:]
        return apply

    def finish(self,index):
        assert sorted(self.parts)==list(range(6)) and self.request is not None
        backward=envelope(self.frame_id,0,self.pose)+self.request
        forward=envelope(self.frame_id,1,self.pose)+struct.pack('<4sI',b'CCW1',6)+b''.join(self.parts[i] for i in range(6))
        rows=[]
        for name,packet in [('demand',backward),('response',forward)]:
            p=self.output/('%06d_%s.bin'%(index,name));p.write_bytes(packet)
            rows.append(dict(direction=name,file='wire/'+p.name,bytes=len(packet),sha256=hashlib.sha256(packet).hexdigest()))
        return dict(packets=rows,demand_bytes=len(backward),response_bytes=len(forward),total_bytes=len(backward)+len(forward),
            envelope_bytes=160,late_boxes=0,late_section_bytes=0,sections=[self.details[i] for i in range(3)])

    def close(self):
        for handle in self.handles:handle.remove()


def self_test():
    rng=np.random.RandomState(303)
    for steps in (8,32):
        for counts in [np.zeros((4,7,9)),rng.randint(0,steps+1,(4,7,9))]:
            scale=np.array([.01,.5,1.,8.],dtype='f4')
            stage,t,got,s=decode_counts(encode_counts(2,counts,scale,steps))
            assert stage==2 and t==steps and np.array_equal(got,counts) and np.array_equal(s,scale)
    for x in [np.zeros((7,9)),rng.randint(0,2,(7,9)),rng.rand(7,9).astype('f4')]:
        stage,got=decode_mask(encode_mask(1,x));assert stage==1 and np.array_equal(got,x.astype('f4'))
    print('Sparse integer count/scale and binary/continuous mask round trips passed')


if __name__=='__main__':self_test()
