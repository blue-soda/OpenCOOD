"""Fixed-wire small/wide BEV main comparison; bounded audit/train/finalize."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from opencood.tools.train_e3dsnn_single import write_json,checked_forward
from opencood.tools import train_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.e3dsnn_snapshot_capacity import E3dsnnSnapshotCapacity
from opencood.utils.e3dsnn_capacity import widen_backend

REPO=Path(__file__).resolve().parents[2]
SOURCE_SHA='0add747f31daf1c34315621990ef1b9424e2606f2cb60fa23cc3f396bf9e6f9e'
SEED=20260929
BUDGETS=(0,2048,2304,4096)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def prepare(root,source_config,source_checkpoint):
    if sha(source_checkpoint)!=SOURCE_SHA:raise ValueError('Wrong fixed2k epoch6 source')
    root.mkdir(exist_ok=False,parents=True)
    cfg=yaml.safe_load(Path(source_config).read_text());cfg['train_params'].update(batch_size=2,epoches=10)
    cfg['optimizer']['lr']=5e-5;cfg['lr_scheduler'].update(step_size=[7,9],gamma=.1)
    cfg['model']['core_method']='e3dsnn_snapshot_capacity'
    cfg['model']['args']['snapshot_joint']['train_budgets']=[2304]
    cfg['model']['args']['snapshot'].update(budget_bytes=2304,packet_directory=None)
    cfg['model']['args']['capacity']={'bev_width':1}
    cfg['model']['args']['message_dropout']=.1
    cfg['model']['args']['snapshot_joint']['ego_aux_weight']=.5
    small=E3dsnnSnapshotCapacity(cfg['model']['args'])
    small.load_state_dict(torch.load(source_checkpoint,map_location='cpu'),strict=True)
    source_state={k:v.clone() for k,v in small.state_dict().items()}
    hashes={};counts={}
    for arm,width in [('c0_small',1),('c1_wide',2)]:
        folder=root/arm;folder.mkdir();c=copy.deepcopy(cfg);c['model']['args']['capacity']['bev_width']=width
        if width==2:widen_backend(small)
        initial=small.state_dict();torch.save(initial,str(folder/'initial.pth'))
        test=E3dsnnSnapshotCapacity(c['model']['args']);test.load_state_dict(initial,strict=True)
        for k,v in initial.items():
            if not k.startswith(('backbone_2d.','cls_head.','reg_head.','dir_head.')):
                assert torch.equal(v,source_state[k]),k
        counts[arm]=dict(total=sum(p.numel() for p in test.parameters()),trainable=sum(p.numel() for p in test.parameters() if p.requires_grad))
        for name,budget in [('train',2304)]+[('b'+str(x),x) for x in BUDGETS]:
            v=copy.deepcopy(c);v['model']['args']['snapshot']['budget_bytes']=budget
            if name!='train':v['model']['args']['snapshot']['packet_directory']=str(folder/name/'packets')
            p=folder/(name+'.yaml');p.write_text(yaml.safe_dump(v));hashes[str(p.relative_to(root))]=sha(p)
        hashes[str((folder/'initial.pth').relative_to(root))]=sha(folder/'initial.pth')
        del test
    assert counts['c0_small']['total']==3120631 and counts['c1_wide']['total']==6556919
    write_json(root/'protocol.json',dict(source=str(source_checkpoint),source_sha256=SOURCE_SHA,source_config_sha256=sha(source_config),
       seed=SEED,primary_budget=2304,final_budgets=BUDGETS,epochs=10,counts=counts,hashes=hashes,
       dataset_hashes={k:sha(cfg[k]) for k in ('root_dir','validate_dir','data_manifest')},
       loss='Ldet(2304)+0.5 Ldet(0); dropout .1; frozen sender/receiver/BN',
       initialization='twofold channel duplication, downstream sum preservation and zero-sum 1e-3 symmetry break',
       no_hard_parameter_ceiling=True,priority='AP versus actual bytes; larger models or architecture changes allowed after evidence',
       selection='best full validation AP70 at 2304 bytes including inherited epoch0',
       gpu_memory_fraction=.14,training_timeout_seconds=345600,evaluation_timeout_seconds=28800))


def audit(root,arm):
    from opencood.data_utils.datasets import build_dataset
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(.14);torch.set_num_threads(1)
    folder=root/arm;cfg=load_yaml(str(folder/'train.yaml'));data=build_dataset(cfg,train=False);train=build_dataset(cfg,train=True)
    assert len(data)==1738 and len(train)==4678
    model=train_utils.create_model(cfg).cuda();model.load_state_dict(torch.load(str(folder/'initial.pth'),map_location='cpu'),strict=True);model.eval()
    ref_cfg=copy.deepcopy(cfg);ref_cfg['model']['args']['capacity']['bev_width']=1
    reference=train_utils.create_model(ref_cfg).cuda();reference.load_state_dict(torch.load(str(root/'c0_small/initial.pth'),map_location='cpu'),strict=True);reference.eval()
    records=[]
    with torch.no_grad():
        for index in (0,579,1158,1737):
            batch=train_utils.to_device(data.collate_batch_test([data[index]]),torch.device('cuda'))
            model.snapshot_budget=reference.snapshot_budget=2304
            actual=model(batch['ego']);original=reference(batch['ego'])
            diffs={k:float((actual[k]-original[k]).abs().max()) for k in ('psm','rm','dm')}
            assert all(torch.allclose(actual[k],original[k],atol=1e-4,rtol=2e-5) for k in diffs),diffs
            assert actual['packet_bytes']==original['packet_bytes'] and max(actual['packet_bytes'])<=2304
            model.packet_roundtrip=True;packets=[];model._snapshot_packet_observer=lambda fid,blob:packets.append(hashlib.sha256(blob).hexdigest())
            wire=model(batch['ego']);reference.packet_roundtrip=True;ref_packets=[];reference._snapshot_packet_observer=lambda fid,blob:ref_packets.append(hashlib.sha256(blob).hexdigest())
            reference(batch['ego']);assert packets==ref_packets
            assert all(torch.equal(actual[k],wire[k]) for k in diffs)
            records.append(dict(index=index,bytes=wire['packet_bytes'],packet_sha256=packets,initial_max_abs=diffs,direct_wire_exact=True))
            model.packet_roundtrip=reference.packet_roundtrip=False;model._snapshot_packet_observer=reference._snapshot_packet_observer=None
        model.snapshot_budget=0;zero=model(batch['ego']);model.fusion_mode='ego';ego=model(batch['ego']);model.fusion_mode='residual'
        assert all(torch.equal(zero[k],ego[k]) for k in ('psm','rm','dm'))
    del reference,batch,original,actual,wire,zero,ego
    torch.cuda.empty_cache();model.snapshot_budget=2304;model.train();fixed=model.fixed_state_digest()
    batch=train_utils.to_device(train.collate_batch_train([train[0],train[1]]),torch.device('cuda'))
    criterion=train_utils.create_loss(cfg).cuda();optimizer=train_utils.setup_optimizer(cfg,model)
    optimizer.zero_grad(set_to_none=True);output,loss=checked_forward(model,criterion,batch);loss.backward()
    grads={prefix:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
           for prefix in ('backbone_2d.','cls_head.','reg_head.','dir_head.')}
    assert all(np.isfinite(x) and x>0 for x in grads.values()),grads
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
    assert model.fixed_state_digest()==fixed
    write_json(folder/'audit.json',dict(passed=True,records=records,gradients=grads,loss=float(loss.detach()),zero_equals_ego=True,
          fixed_state_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_audit_updates_discarded=True))


def coordinate(root,arm):
    folder=root/arm;protocol=read(root/'protocol.json')
    for name,digest in protocol['hashes'].items():assert sha(root/name)==digest
    cfg=load_yaml(str(folder/'train.yaml'))
    for key,digest in protocol['dataset_hashes'].items():assert sha(cfg[key])==digest
    def run(stage,command,timeout):
        write_json(folder/'status.json',dict(state='running',stage=stage,unix=time.time()))
        with (folder/(stage+'.log')).open('xb') as log:
            subprocess.run(['timeout','--kill-after=30',str(timeout)]+command,cwd=str(REPO),env=os.environ.copy(),
                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,start_new_session=True)
    try:
        run('audit',[sys.executable,'-u','-m',__spec__.name,'--mode','audit','--root',str(root),'--arm',arm],1800)
        assert read(folder/'audit.json')['passed']
        base=[sys.executable,'-u','-m','opencood.tools.train_e3dsnn_single','--workers','2','--seed',str(SEED)]
        run('train',base+['--mode','train','--config',str(folder/'train.yaml'),'--checkpoint',str(folder/'initial.pth'),
            '--output',str(folder/'train'),'--epochs','10','--selection-iou','0.7','--retain-initial-best'],protocol['training_timeout_seconds'])
        result={}
        from opencood.tools.supervise_e3dsnn_snapshot_eval import verify_budget
        for budget in BUDGETS:
            name='b'+str(budget);out=folder/name
            run(name,base+['--mode','eval','--config',str(folder/(name+'.yaml')),'--checkpoint',str(folder/'train/best_ap70.pth'),
                '--output',str(out),'--packet-roundtrip','--eval-3d','--save-predictions','--profile'],protocol['evaluation_timeout_seconds'])
            run(name+'_replay',[sys.executable,'-m','opencood.tools.replay_e3dsnn_predictions','--evaluation-dir',str(out)],3600)
            result[name]=verify_budget(folder,budget)
        write_json(folder/'summary.json',result);write_json(folder/'status.json',dict(state='complete'))
    except Exception:
        write_json(folder/'status.json',dict(state='failed',error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','audit','coordinate'],required=True);p.add_argument('--root',required=True)
    p.add_argument('--arm',choices=['c0_small','c1_wide']);p.add_argument('--source-config');p.add_argument('--source-checkpoint')
    a=p.parse_args();root=Path(a.root)
    if a.mode=='prepare':prepare(root,a.source_config,a.source_checkpoint)
    elif a.mode=='audit':audit(root,a.arm)
    else:coordinate(root,a.arm)
