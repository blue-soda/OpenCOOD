"""Register equal-capacity local/message context arms from the original F0."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import torch
import yaml
from opencood.models.fuse_modules.e3dsnn_context_gain import SnapshotContextGain
from opencood.tools.prepare_cp_budget_round import F0_SHA, REPO, dump, job, eval_jobs
from opencood.tools.run_registered_research import sha


def prepare(a):
    source=Path(a.checkpoint); assert sha(source)==F0_SHA
    cfg=yaml.safe_load(Path(a.config).read_text())
    assert cfg['model']['core_method']=='e3dsnn_snapshot_frontend'
    manifest=json.loads(Path(a.manifest).read_text())
    assert manifest['data_root']==a.data_root, 'Audit the destination data root before registering'
    assert [manifest['splits'][s]['valid_count'] for s in ('train','val')]==[4678,1738]
    root=Path(a.root); root.mkdir(parents=True, exist_ok=False)
    cfg['data_dir']=a.data_root
    for k,f in [('root_dir','train.json'),('validate_dir','val.json'),('test_dir','val.json')]:
        cfg[k]=str(Path(a.data_root)/f)
    cfg['data_manifest']=a.manifest
    cfg['model']['core_method']='e3dsnn_snapshot_context'
    cfg['model']['args']['snapshot_context']=dict(condition=a.condition)
    cfg['model']['args']['snapshot_joint']['train_budgets']=[2304]
    cfg['model']['args']['snapshot'].update(budget_bytes=2304, packet_directory=None)
    cfg['train_params'].update(epoches=6,batch_size=2)
    cfg['optimizer']['lr']=2e-5
    cfg['lr_scheduler'].update(step_size=[4,5], gamma=.1)
    state=torch.load(str(source), map_location='cpu')
    torch.manual_seed(20260929)
    gate=SnapshotContextGain(a.condition)
    additions={'joint_receiver.snapshot_context_gate.'+k:v for k,v in gate.state_dict().items()}
    assert not set(additions).intersection(state)
    state.update(additions); torch.save(state,str(root/'initial.pth'))
    (root/'train.yaml').write_text(yaml.safe_dump(cfg))
    jobs=[job('smoke','opencood.tools.audit_cp_context',
        ['--config',root/'train.yaml','--checkpoint',root/'initial.pth','--parent',source,
         '--output',root/'audit.json'],3600,[root/'audit.json']),
        job('train','opencood.tools.train_e3dsnn_single',
        ['--mode','train','--config',root/'train.yaml','--checkpoint',root/'initial.pth',
         '--output',root/'train','--epochs',6,'--workers',2,'--seed',20260929,
         '--selection-iou',.7,'--retain-initial-best'],345600,
        [root/'train/status.json',root/'train/best_ap70.pth'])]
    for b in (2304,0,4096):
        c=copy.deepcopy(cfg)
        c['model']['args']['snapshot'].update(budget_bytes=b,packet_directory=str(root/('b'+str(b))/'packets'))
        path=root/('b'+str(b)+'.yaml');path.write_text(yaml.safe_dump(c))
        jobs+=eval_jobs(root,b,path,root/'train/best_ap70.pth',a.reference)
    inputs=[source,Path(a.config),root/'initial.pth']+list(root.glob('*.yaml'))
    inputs += [Path(cfg[k]) for k in ('root_dir','validate_dir','data_manifest')]
    reference=Path(a.reference)
    inputs += [reference/'common_gt_manifest.json',reference/'common_val.json']
    for row in json.loads((reference/'common_gt_manifest.json').read_text())['frames']:
        p=reference/'common_gt'/(row['frame_id']+'.npy'); assert sha(p)==row['sha256'];inputs.append(p)
    for folder in a.dependency_paths:
        inputs += [p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inputs += list((REPO/'opencood/utils').glob('box_overlaps*.so'))
    protocol=dict(repo=str(REPO),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        input_hashes={str(p):sha(p) for p in inputs},dependency_paths=a.dependency_paths,jobs=jobs,
        condition=a.condition,seed=20260929,source=str(source),source_sha256=F0_SHA,primary_budget=2304,
        hypothesis='Does received-message context improve equal-capacity gating relative to duplicated local membrane?',
        selection='Full native val AP70 at 2304B including epoch0, then fixed common CPU GT replay at 0/2304/4096B',
        training='Six epochs, F0 original best, full frontend plus new gate, lr2e-5 decay4/5, Ldet+.5Lego, dropout .1, frozen BN',
        exit_rule='Report both arms and epoch0; no post-hoc budget selection or extended training',
        scope='Keep SNN encoder, IF8 count message, warp, membrane Count4 and ANN BEV; no new communication',
        boundary='Single-seed exploration; independently trained sender contents may differ; no energy or mechanism claim')
    dump(root/'protocol.json',protocol)
    print(json.dumps(dict(root=str(root),condition=a.condition,jobs=len(jobs),initial_sha256=sha(root/'initial.pth'))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','config','checkpoint','data-root','manifest','reference'):p.add_argument('--'+key,required=True)
    p.add_argument('--condition',choices=('local','message'),required=True)
    p.add_argument('--dependency-paths',nargs='*',default=[])
    prepare(p.parse_args())
