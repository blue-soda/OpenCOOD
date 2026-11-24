"""Check an exact storage-efficient previous-target control using the already retained block accumulators."""
import argparse
from collections import deque
import copy
import json
from pathlib import Path

import torch
from opencood.tools import train_int_spike_pyramid as fit
from opencood.tools.train_int_anchor_refresh import RecoveryPyramid
from opencood.tools.train_int_full_dair_pair import full_rows
from opencood.tools.check_int_silent_trajectory import scene_windows
from opencood.data_utils.int_v2v_ego_stream import v2v_rows
from opencood.models.sub_modules.int_planar_window import attach_window


def bytes_in(tensor):
    return tensor.numel()*tensor.element_size()


class PreviousTargetStore:
    def __init__(self,net):
        self.positions = self.vectors = None
        self.restored_elements = 0
        net.input_sender.register_forward_pre_hook(self.restore_input)
        net.input_sender.register_forward_hook(self.pack_input)
        for stage in net.stages:
            for block in stage:
                block.register_forward_pre_hook(self.restore_block)
                for sender in (block.sender1,block.sender2):
                    sender.register_forward_hook(self.release)

    def restore_input(self,sender,args):
        target,prior=args
        if torch.is_tensor(prior):
            _,channels,height,width = target.shape
            flat=target.new_zeros(height*width,channels)
            flat.index_copy_(0,self.positions.long(),self.vectors)
            sender.sending_reference=flat.t().reshape_as(target)

    def pack_input(self,sender,args,output):
        target=args[0]
        assert target.size(0)==1 and target.size(2)*target.size(3)<2**31
        vectors=target[0].flatten(1).t()
        indices=vectors.ne(0).any(dim=1).nonzero().flatten()
        self.positions=indices.int()
        self.vectors=vectors.index_select(0,indices)
        sender.sending_reference=None

    def restore_block(self,block,args):
        _,state=args
        if state is not None:
            block.sender1.sending_reference=state[0].relu()
            block.sender2.sending_reference=state[2].relu()
            self.restored_elements += state[0].numel()+state[2].numel()

    def release(self,sender,args,output):
        sender.sending_reference=None


@torch.no_grad()
def run(args):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    cfg=fit.load_yaml(str(args.root/'inputs/config.yaml'))
    cfg['data_augment']=[]
    source=fit.build_source(args.root/'inputs',cfg,device=device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    assert checkpoint['contract']['adjacent_target'] and checkpoint['contract']['seed']==303
    assert not checkpoint['contract'].get('shift_codes') and not checkpoint['contract'].get('training_payload')
    explicit=RecoveryPyramid(source.core,args.root/'initial.pt','fine_temporal').to(device).eval()
    explicit.load_state_dict(checkpoint['net'])
    for sender in explicit.senders:
        sender.adjacent_target=True
    compact=copy.deepcopy(explicit)
    store=PreviousTargetStore(compact)
    pending=deque()
    def record(sender,inputs,output):
        pending.append(output[0])
    def compare(sender,inputs,output):
        assert torch.equal(pending.popleft(),output[0])
    for sender in explicit.senders: sender.register_forward_hook(record)
    for sender in compact.senders: sender.register_forward_hook(compare)
    if args.domain=='dair':
        manifest=args.root/'sequence_manifest.json'
        validation=full_rows(manifest,args.data,'val')
    else:
        manifest=args.root/'formal_splits/v2v_official_deduplicated.jsonl'
        validation=v2v_rows(manifest,'val',args.data)
    assert fit.digest(manifest)==checkpoint['contract']['manifest']
    rows=attach_window(scene_windows(validation),False)
    receipts=[]
    state_a=state_b=None
    for row,sample in zip(rows,fit.loader(rows,cfg,0,303)[1]):
        assert sample['meta']['frame']==row['frame']
        if row['reset_before']: state_a=state_b=None
        ego=fit.to_device(sample['ego'],device)
        spatial=fit.spatial_input(source,ego)
        old=store.restored_elements
        a,state_a,_=explicit(spatial,source.core,state_a,ordinary=row['window_ordinary'])
        assert len(pending)==len(explicit.senders)
        b,state_b,_=compact(spatial,source.core,state_b,ordinary=row['window_ordinary'])
        assert not pending and all(torch.equal(a[k],b[k]) for k in a)
        assert torch.equal(state_a[0],state_b[0]) and set(state_a[1])==set(state_b[1])
        for k in state_a[1]:
            assert all(torch.equal(x,y) for x,y in zip(state_a[1][k],state_b[1][k]))
        assert all(s.sending_reference is None for s in compact.senders)
        receipts.append(dict(scene=row['scene'],frame=row['frame'],ordinary=row['window_ordinary'],
                             extra_bytes=bytes_in(store.positions)+bytes_in(store.vectors),
                             explicit_bytes=sum(bytes_in(s.sending_reference) for s in explicit.senders),
                             rectified_elements=store.restored_elements-old, stored_sites=store.positions.numel()))
    assert len(receipts)==len(rows)>0
    with args.out.open('x') as handle:
        json.dump(dict(rows=receipts,parent=fit.digest(args.checkpoint),manifest=fit.digest(manifest),
                       senders=len(explicit.senders),scope='Additional persistent tensor payload after each scan; excludes allocator, scratch, reads and writes. All sender messages, retained block states and unmodified readout logits compared exactly.'),handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','data','checkpoint','out'): parser.add_argument(name,type=Path)
    parser.add_argument('domain',choices=('dair','v2v'))
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    run(parser.parse_args())
