"""CoSDH real-wire parity gate. Diagnostic overhead forbids speed claims."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from compute_harness import fingerprint, run_pass, selected_indices
from compute_infocom_adapter import InfoComAdapter, sha, write


class CoSDHAdapter(InfoComAdapter):
    def reset(self, frame_id):
        self.frame_id=frame_id;self.messages=None

    def infer(self, batch):
        from opencood.tools import inference_utils
        self.wire.begin(batch,self.frame_id)
        result=inference_utils.inference_late_fusion(batch,self.model,self.dataset)
        # Envelopes and concatenation are actual sender work, included in time.
        self.messages=self.wire.packets()
        return result

    def packets(self):
        if self.messages is None:raise ValueError('No fresh actual packets')
        return self.messages


def smoke(model,dataset,wire,ids,reference,output,meta,gpu,loader):
    if wire is None or len(ids)!=1738:raise ValueError('Full common split and wire required')
    refmeta=json.loads((reference/'provenance.json').read_text())
    for key in ('checkpoint_sha256','source_config_sha256','split_sha256','wire_codec','fusion_method','seed'):
        if refmeta.get(key)!=meta.get(key):raise ValueError('Reference protocol differs: '+key)
    if refmeta['overrides'].get('comm_range')!=meta['overrides'].get('comm_range'):
        raise ValueError('Participant range differs')
    rows=[json.loads(x) for x in (reference/'frames.jsonl').read_text().splitlines()]
    if [r['frame_id'] for r in rows]!=ids:raise ValueError('Reference ID order differs')
    all_indices=selected_indices();indices=[all_indices[x] for x in (0,21,42,63)];refs={}
    for i in indices:
        row=rows[i];p=reference/row['file']
        if sha(p)!=row['sha256']:raise ValueError('Prediction SHA mismatch')
        with np.load(str(p)) as z:arrays=dict(boxes=z['pred'],scores=z['score'])
        packets=[]
        for item in row['wire']['packets']:
            p=reference/item['file'];blob=p.read_bytes()
            if sha(p)!=item['sha256'] or len(blob)!=item['bytes']:raise ValueError('Packet mismatch')
            packets.append((item['direction'],blob))
        refs[str(ids[i])]=fingerprint(ids[i],arrays,packets)
        if refs[str(ids[i])]['total_bytes']!=row['wire']['total_bytes']:raise ValueError('Bytes mismatch')
    def resource():
        def query(args):return subprocess.check_output(['nvidia-smi']+args).decode().strip()
        return dict(gpu=query(['-i',str(gpu),'--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader']),
                    compute_processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),
                    own_pid=os.getpid(),cpu_load=list(os.getloadavg()),torch_threads=torch.get_num_threads())
    protocol=dict(smoke_only=True,not_a_performance_result=True,indices=indices,ids=[ids[i] for i in indices],
                  checkpoint_sha256=meta['checkpoint_sha256'],reference_manifest_sha256=sha(reference/'frames.jsonl'),
                  warmup=20,scopes=['ready_cpu_batch_to_cpu_boxes','device_batch_to_cpu_boxes'],
                  real_codec=True,packet_file_io_inside_timing=False,envelope_assembly_inside_timing=True,
                  retains_redundant_diagnostic_decode_and_warp=True,eligible_for_main_speed_table=False,
                  preprocessing='full ordered RNG replay, four cached CPU batches',model_or_message_cache=False,
                  script_sha256=sha(__file__),wire_sha256=sha(Path(__file__).with_name('cosdh_wire.py')))
    write(output/'compute_protocol.json',protocol)
    prepared={}
    for i,batch in enumerate(loader):
        if i in indices:prepared[i]=batch
        if i%250==0:write(output/'status.json',dict(state='preparing_reference_inputs',processed=i+1))
        if i==indices[-1]:break
    if set(prepared)!=set(indices):raise ValueError('Incomplete CPU replay')
    adapter=CoSDHAdapter(model,dataset,wire,ids,prepared);results=[]
    with torch.no_grad():
        for scope in protocol['scopes']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            result=run_pass(adapter,indices,refs,scope=scope,synchronize=torch.cuda.synchronize,
                            resource_snapshot=resource,warmup=20)
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(result);write(output/'compute_results_so_far.json',results)
    write(output/'compute_summary.json',dict(passed=True,protocol=protocol,results=results,
          exact_reference_prediction_and_packet_content=True,unique_frames=4,timed_calls=8,warmup_calls=40,
          no_full_AP_or_speed_claim=True))
    write(output/'status.json',dict(finished=True,compute_smoke_only=True,unique_frames=4))
