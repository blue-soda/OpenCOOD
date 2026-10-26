"""Real InfoCom adapter gate; four fixed frames, no AP or speed conclusions."""
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import numpy as np
import torch
from compute_harness import fingerprint, run_pass, measurement_indices, host_metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


class InfoComAdapter:
    def __init__(self, model, dataset, wire, ids, prepared, rng_states=None):
        self.model=model;self.dataset=dataset;self.wire=wire;self.ids=ids;self.frame_id=None
        self.prepared=prepared
        self.rng_states=rng_states

    def prepare_cpu(self, index):
        # Official evaluation also shuffles points. Reusing the prepared CPU
        # batch fixes voxel membership across warmup and timed repetitions.
        batch=self.prepared[index]
        if batch is None or int(batch['ego']['sample_idx']) != index:
            raise ValueError('Frame identity mismatch')
        return str(self.ids[index]),batch

    def reset(self, frame_id):
        self.frame_id=frame_id;self.wire.messages=[]
        if self.rng_states is not None:
            cpu,cuda=self.rng_states[str(frame_id)]
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda)

    def to_device(self, batch):
        from opencood.tools import train_utils
        return train_utils.to_device(batch,torch.device('cuda'))

    def infer(self, batch):
        from opencood.tools import inference_utils
        self.wire.begin(batch,self.frame_id)
        return inference_utils.inference_intermediate_fusion(batch,self.model,self.dataset)

    def to_cpu(self, result):
        def convert(tensor,shape):
            return np.empty(shape,np.float32) if tensor is None else tensor.detach().cpu().numpy()
        return dict(boxes=convert(result['pred_box_tensor'],(0,8,3)),
                    scores=convert(result['pred_score'],(0,)))

    def packets(self):
        # WireAudit.hook executed the exact codec and substituted its decoded
        # messages into the real receiver. finish() would only write files.
        return [('road_to_vehicle_sender%d'%r['sender'],p) for r,p in self.wire.messages]


def smoke(model,dataset,wire,ids,reference,output,meta,gpu,loader,count=4,repeat=0,diagnostic=False):
    if wire is None or len(ids)!=1738:
        raise ValueError('Need real wire and the full common ID list')
    refmeta=json.loads((reference/'summary.json').read_text())
    if refmeta['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Not the same checkpoint as the immutable reference')
    for key in ('split_sha256','participant_range_metres','wire_codec','mask_ratio_override'):
        if refmeta.get(key)!=meta.get(key):raise ValueError('Reference protocol differs: '+key)
    index=json.loads((reference/'predictions_index.json').read_text())
    if [x['frame_id'] for x in index]!=ids:raise ValueError('Reference ID order differs')
    indices=measurement_indices(count)
    refs={}
    for i in indices:
        row=index[i];p=reference/row['file']
        if sha(p)!=row['sha256']:raise ValueError('Reference prediction SHA mismatch')
        with np.load(str(p)) as z:arrays=dict(boxes=z['pred'],scores=z['score'])
        packets=[]
        for message in row['wire']['messages']:
            p=reference/message['file'];blob=p.read_bytes()
            if sha(p)!=message['sha256']:raise ValueError('Reference packet SHA mismatch')
            packets.append(('road_to_vehicle_sender%d'%message['sender'],blob))
        refs[str(ids[i])]=fingerprint(ids[i],arrays,packets)
        if refs[str(ids[i])]['total_bytes']!=row['wire']['total_bytes']:
            raise ValueError('Reference total bytes differ')
    def resource():
        def query(args):
            return subprocess.check_output(['nvidia-smi']+args).decode().strip()
        return dict(gpu=query(['-i',str(gpu),'--query-gpu=uuid,name,driver_version,memory.total,memory.free,utilization.gpu',
                               '--format=csv,noheader']),
                    compute_processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),
                    own_pid=os.getpid(),cpu_load=list(os.getloadavg()),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads())
    protocol=dict(smoke_only=count==4,not_a_performance_result=count==4,repeat=repeat,host=host_metadata(),indices=indices,ids=[ids[i] for i in indices],
                  checkpoint_sha256=meta['checkpoint_sha256'],reference_summary_sha256=sha(reference/'summary.json'),
                  reference_index_sha256=sha(reference/'predictions_index.json'),warmup=20,
                  scopes=['ready_cpu_batch_to_cpu_boxes','device_batch_to_cpu_boxes'],
                  real_codec=True,packet_file_io_inside_timing=False,reference_validation_outside_timing=True,
                  retains_reference_codec_roundtrip_checks=True,model_weight_updates=False,
                  script_sha256=sha(__file__),harness_sha256=sha(Path(__file__).with_name('compute_harness.py')))
    write(output/'compute_protocol.json',protocol)
    # Official topk_mask uses CUDA randperm for ties even in eval(). Reproduce
    # ALL preceding forwards plus the original first-frame warmups, retaining
    # only CPU inputs and pre-forward RNG states. No output/message is cached.
    from opencood.tools import inference_utils, train_utils
    prepared={};rng_states={}
    reference_warmup=int(refmeta.get('warmup_passes',3))
    with torch.no_grad():
        for i,batch in enumerate(loader):
            device_batch=train_utils.to_device(batch,torch.device('cuda'))
            fid=str(ids[i])
            if i==0:
                states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
                for _ in range(reference_warmup):
                    wire.begin(device_batch,fid)
                    inference_utils.inference_intermediate_fusion(device_batch,model,dataset)
                # evaluate_infocom restores the scored pass RNG after warmup.
                random.setstate(states[0]);np.random.set_state(states[1])
                torch.set_rng_state(states[2]);torch.cuda.set_rng_state_all(states[3])
            if i in indices:
                prepared[i]=batch
                rng_states[fid]=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            wire.begin(device_batch,fid)
            result=inference_utils.inference_intermediate_fusion(device_batch,model,dataset)
            if i in indices:
                probe=InfoComAdapter(model,dataset,wire,ids,prepared)
                actual=fingerprint(fid,probe.to_cpu(result),probe.packets())
                if actual!=refs[fid]:
                    write(output/'reference_replay_mismatch.json',dict(index=i,frame_id=fid,actual=actual,expected=refs[fid]))
                    raise ValueError('Full ordered RNG replay differs from immutable reference at '+fid)
            if i%250==0:write(output/'status.json',dict(state='replaying_reference_rng',processed=i+1))
            if i==indices[-1]:break
    if set(prepared)!=set(indices):raise ValueError('Incomplete reference input replay')
    protocol.update(preprocessing='full ordered input and model RNG replay, including original warmups',
                    rng_replay=dict(reference_warmup=reference_warmup,selected_states=len(rng_states),
                                    reference_warmup_rng_restored=True,restore_before_timing=True,
                                    reason='official stochastic top-k tie breaking'),
                    fresh_inputs_for_each_repeat=False,model_or_message_cache=False)
    write(output/'compute_protocol.json',protocol)
    adapter=InfoComAdapter(model,dataset,wire,ids,prepared,rng_states);results=[]
    if diagnostic:
        from diagnose_snapshot_compute import diagnose
        protocol.update(diagnostic_only=True,not_a_performance_result=True)
        write(output/'compute_protocol.json',protocol)
        diagnose(adapter,indices,refs,reference,output)
        write(output/'status.json',dict(finished=True,diagnostic_only=True,unique_frames=len(indices)))
        return
    with torch.no_grad():
        for scope in protocol['scopes']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            result=run_pass(adapter,indices,refs,scope=scope,synchronize=torch.cuda.synchronize,
                            resource_snapshot=resource,warmup=20,repeat=repeat)
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(result);write(output/'compute_results_so_far.json',results)
    write(output/'compute_summary.json',dict(passed=True,protocol=protocol,results=results,
          exact_reference_prediction_and_packet_content=True,unique_frames=count,timed_calls=2*count,warmup_calls=40,
          no_full_AP_claim=True,complete_three_repeat_comparison=False))
    write(output/'status.json',dict(finished=True,compute_smoke_only=count==4,unique_frames=count))
    print(json.dumps(dict(passed=True,compute_smoke_only=count==4,frames=count,timed_calls=2*count)),flush=True)
