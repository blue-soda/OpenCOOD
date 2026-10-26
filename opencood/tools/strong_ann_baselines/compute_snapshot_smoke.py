"""Four fixed-frame Scheme-2 CPU-to-box/wire equivalence gate, not a benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import traceback
import numpy as np
import torch
from opencood.tools.strong_ann_baselines.compute_harness import fingerprint,run_pass,measurement_indices,host_metadata


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)


def configure_backend(model, mode):
    """Opt-in numerical diagnostic, never a change to the trained activation."""
    from spconv.pytorch.conv import SparseConvolution
    from spconv.core import ConvAlgo
    from spconv.constants import ALL_WEIGHT_IS_KRSC
    layers=[(name,m) for name,m in model.named_modules() if isinstance(m,SparseConvolution)]
    before={name:str(m.algo) for name,m in layers}
    if mode=='native_deterministic':
        # Changing algorithms is safe only with the same stored KRSC layout.
        # Every layer sharing an indice key must use the same algorithm.
        if not ALL_WEIGHT_IS_KRSC:raise RuntimeError('Native diagnostic requires shared KRSC weight layout')
        if not layers:raise RuntimeError('No sparse convolutions found')
        for name,m in layers:m.algo=ConvAlgo.Native
    return dict(mode=mode,spconv_before=before,spconv_after={name:str(m.algo) for name,m in layers},
                all_weight_is_krsc=ALL_WEIGHT_IS_KRSC,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                spconv_determinism_guaranteed=False)


class Adapter:
    def __init__(self,model,data,prepared):
        self.model=model;self.data=data;self.prepared=prepared;self.messages=[];self.fid=None
        model._snapshot_packet_observer=self.receive
    def receive(self,fid,packet):
        if str(fid)!=self.fid:raise ValueError('Wire frame identity changed')
        self.messages.append(('road_to_vehicle',packet))
    def prepare_cpu(self,index):return str(self.data.data[index]),self.prepared[index]
    def reset(self,fid):self.fid=fid;self.messages=[]
    def to_device(self,batch):
        from opencood.tools import train_utils
        return train_utils.to_device(batch,torch.device('cuda'))
    def infer(self,batch):
        output=self.model(batch['ego'])
        return self.data.post_process_no_fusion(batch,{'ego':output})
    def to_cpu(self,result):
        boxes,scores,_=result
        return dict(boxes=np.empty((0,8,3),np.float32) if boxes is None else boxes.detach().cpu().numpy(),
                    scores=np.empty((0,),np.float32) if scores is None else scores.detach().cpu().numpy())
    def packets(self):return self.messages


def execute(a):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    root=Path(a.output);reference=Path(a.reference);checkpoint=Path(a.checkpoint)
    manifests=list(reference.glob('manifest_*.json'))
    if len(manifests)!=1:raise ValueError('Ambiguous reference provenance')
    manifest=read(manifests[0])
    if sha(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Wrong checkpoint')
    if sha(a.config)!=manifest['config_sha256']:raise ValueError('Wrong reference config')
    if a.backend=='native_deterministic':
        if not a.diagnose:raise ValueError('Backend experiment is diagnostic only; historical timing gate unchanged')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
    cfg=load_yaml(a.config)
    if cfg['model']['core_method']!='e3dsnn_snapshot_joint':raise ValueError('Expected Scheme-2 joint model')
    if cfg['model']['args']['snapshot']['budget_bytes']!=2048:raise ValueError('Registered 2KiB point only')
    cfg['model']['args']['snapshot']['packet_directory']=None
    data=build_dataset(cfg,train=False)
    if len(data)!=1738:raise ValueError('Wrong paired validation')
    model=train_utils.create_model(cfg).cuda();model.load_state_dict(torch.load(str(checkpoint),map_location='cpu'),strict=True)
    backend=configure_backend(model,a.backend)
    model.eval();model.requires_grad_(False);model.packet_roundtrip=True
    indices=measurement_indices(a.compute_count)
    index=read(reference/'predictions/index.json')['frames']
    if [x['frame_id'] for x in index]!=list(data.data):raise ValueError('Reference IDs differ')
    packets=[json.loads(x) for x in (reference/'packets/index.jsonl').read_text().splitlines()]
    if len(packets)!=1738:raise ValueError('Incomplete packet reference')
    refs={};prepared={}
    for i in indices:
        row=index[i];packet=packets[i];p=reference/'predictions'/row['file'];blob=(reference/'packets'/packet['file']).read_bytes()
        if sha(p)!=row['sha256'] or hashlib.sha256(blob).hexdigest()!=packet['sha256']:raise ValueError('Reference SHA mismatch')
        if packet['frame_id']!=row['frame_id']:raise ValueError('Wrong packet frame')
        with np.load(str(p)) as z:arrays=dict(boxes=z['pred_boxes'],scores=z['scores'])
        refs[row['frame_id']]=fingerprint(row['frame_id'],arrays,[('road_to_vehicle',blob)])
        if len(blob)!=row['communication']['total_bytes']:raise ValueError('Reference byte mismatch')
        prepared[i]=data.collate_batch_test([data[i]])
    def resource():
        def query(args):return subprocess.check_output(['nvidia-smi']+args).decode().strip()
        return dict(gpu=query(['-i',a.gpu,'--query-gpu=uuid,name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader']),
                    compute_processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']),
                    own_pid=os.getpid(),cpu_load=list(os.getloadavg()),cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads())
    protocol=dict(smoke_only=a.compute_count==4,not_a_performance_result=a.compute_count==4,repeat=a.compute_repeat,
                  host=host_metadata(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                  indices=indices,ids=[str(data.data[i]) for i in indices],
                  checkpoint_sha256=sha(checkpoint),config_sha256=sha(a.config),config_change='packet_directory=None; observer holds actual packet in memory',
                  reference_index_sha256=sha(reference/'predictions/index.json'),warmup=20,
                  preprocessing='paired validation is deterministic; same selected CPU inputs reused, no message/model cache',
                  scripts={p:sha(Path(__file__).resolve().parents[2]/p) for p in
                      ['tools/strong_ann_baselines/compute_harness.py','tools/strong_ann_baselines/compute_snapshot_smoke.py','models/fuse_modules/e3dsnn_snapshot_adapter.py']},
                  parameters=sum(p.numel() for p in model.parameters()),gpu_memory_fraction=.25,backend=backend)
    write(root/'compute_protocol.json',protocol);adapter=Adapter(model,data,prepared);results=[]
    if a.diagnose:
        from opencood.tools.strong_ann_baselines.diagnose_snapshot_compute import diagnose
        protocol.update(diagnostic_only=True,not_a_performance_result=True)
        write(root/'compute_protocol.json',protocol)
        diagnose(adapter,indices,refs,reference,root)
        write(root/'status.json',dict(finished=True,diagnostic_only=True,unique_frames=len(indices)))
        return
    with torch.no_grad():
        for scope in ['ready_cpu_batch_to_cpu_boxes','device_batch_to_cpu_boxes']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            result=run_pass(adapter,indices,refs,scope=scope,synchronize=torch.cuda.synchronize,resource_snapshot=resource,warmup=20,repeat=a.compute_repeat)
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(result);write(root/'compute_results_so_far.json',results)
    write(root/'compute_summary.json',dict(passed=True,protocol=protocol,results=results,
          exact_reference_prediction_and_packet_content=True,unique_frames=a.compute_count,timed_calls=2*a.compute_count,warmup_calls=40,
          no_full_AP_claim=True,complete_three_repeat_comparison=False))
    write(root/'status.json',dict(finished=True,compute_smoke_only=a.compute_count==4,unique_frames=a.compute_count))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('config','checkpoint','reference','output','gpu'):p.add_argument('--'+n,required=True)
    p.add_argument('--compute-count',type=int,choices=(4,64),default=4)
    p.add_argument('--compute-repeat',type=int,choices=(0,1,2),default=0)
    p.add_argument('--diagnose',action='store_true',help='Two untimed passes with mismatch arrays/packets; no gate relaxation')
    p.add_argument('--backend',choices=('historical','native_deterministic'),default='historical')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    try:execute(a)
    except Exception:
        write(root/'failure.json',dict(passed=False,error=traceback.format_exc(),automatic_retry=False));raise
