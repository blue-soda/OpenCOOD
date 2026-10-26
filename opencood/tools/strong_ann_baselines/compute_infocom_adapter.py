"""Real InfoCom adapter gate; four fixed frames, no AP or speed conclusions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from compute_harness import fingerprint, run_pass, selected_indices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


class InfoComAdapter:
    def __init__(self, model, dataset, wire, ids):
        self.model=model;self.dataset=dataset;self.wire=wire;self.ids=ids;self.frame_id=None

    def prepare_cpu(self, index):
        batch=self.dataset.collate_batch_test([self.dataset[index]])
        if batch is None or int(batch['ego']['sample_idx']) != index:
            raise ValueError('Frame identity mismatch')
        return str(self.ids[index]),batch

    def reset(self, frame_id):
        self.frame_id=frame_id;self.wire.messages=[]

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


def smoke(model,dataset,wire,ids,reference,output,meta,gpu):
    if wire is None or len(ids)!=1738:
        raise ValueError('Need real wire and the full common ID list')
    refmeta=json.loads((reference/'summary.json').read_text())
    if refmeta['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Not the same checkpoint as the immutable reference')
    for key in ('split_sha256','participant_range_metres','wire_codec','mask_ratio_override'):
        if refmeta.get(key)!=meta.get(key):raise ValueError('Reference protocol differs: '+key)
    index=json.loads((reference/'predictions_index.json').read_text())
    if [x['frame_id'] for x in index]!=ids:raise ValueError('Reference ID order differs')
    all_indices=selected_indices();indices=[all_indices[x] for x in (0,21,42,63)]
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
        return dict(gpu=query(['-i',str(gpu),'--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu',
                               '--format=csv,noheader']),
                    compute_processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),
                    own_pid=os.getpid(),cpu_load=list(os.getloadavg()),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads())
    protocol=dict(smoke_only=True,not_a_performance_result=True,indices=indices,ids=[ids[i] for i in indices],
                  checkpoint_sha256=meta['checkpoint_sha256'],reference_summary_sha256=sha(reference/'summary.json'),
                  reference_index_sha256=sha(reference/'predictions_index.json'),warmup=20,
                  scopes=['ready_cpu_batch_to_cpu_boxes','device_batch_to_cpu_boxes'],
                  real_codec=True,packet_file_io_inside_timing=False,reference_validation_outside_timing=True,
                  retains_reference_codec_roundtrip_checks=True,model_weight_updates=False,
                  script_sha256=sha(__file__),harness_sha256=sha(Path(__file__).with_name('compute_harness.py')))
    write(output/'compute_protocol.json',protocol)
    adapter=InfoComAdapter(model,dataset,wire,ids);results=[]
    with torch.no_grad():
        for scope in protocol['scopes']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            result=run_pass(adapter,indices,refs,scope=scope,synchronize=torch.cuda.synchronize,
                            resource_snapshot=resource,warmup=20)
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(result);write(output/'compute_results_so_far.json',results)
    write(output/'compute_summary.json',dict(passed=True,protocol=protocol,results=results,
          exact_reference_prediction_and_packet_content=True,unique_frames=4,timed_calls=8,warmup_calls=40,
          no_full_AP_or_speed_claim=True))
    write(output/'status.json',dict(finished=True,compute_smoke_only=True,unique_frames=4))
    print(json.dumps(dict(passed=True,compute_smoke_only=True,frames=4,timed_calls=8)),flush=True)
