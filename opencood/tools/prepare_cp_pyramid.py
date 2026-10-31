"""Register receiver admission or one arm of the architecture x budget study."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import torch
import yaml
from opencood.models.fuse_modules.e3dsnn_pyramid_current import PyramidCurrent
from opencood.tools.prepare_cp_budget_round import F0_SHA, REPO, dump, job, eval_jobs
from opencood.tools.run_registered_research import sha

MODEL_FILES = ('opencood/models/e3dsnn_snapshot_pyramid.py',
    'opencood/models/fuse_modules/e3dsnn_pyramid_current.py',
    'opencood/models/fuse_modules/e3dsnn_event_stream.py',
    'opencood/utils/cp_optimizer.py', 'opencood/tools/train_e3dsnn_single.py')


def prepare(a):
    source=Path(a.checkpoint);assert sha(source)==F0_SHA
    cfg=yaml.safe_load(Path(a.config).read_text())
    assert cfg['model']['core_method']=='e3dsnn_snapshot_frontend'
    manifest=json.loads(Path(a.manifest).read_text())
    assert manifest['data_root']==a.data_root
    assert [manifest['splits'][s]['valid_count'] for s in ('train','val')]==[4678,1738]
    cfg['data_dir']=a.data_root
    for key,file in [('root_dir','train.json'),('validate_dir','val.json'),('test_dir','val.json')]:
        cfg[key]=str(Path(a.data_root)/file)
    cfg['data_manifest']=a.manifest
    cfg['model']['args']['snapshot_joint']['train_budgets']=[a.budget]
    cfg['model']['args']['snapshot'].update(budget_bytes=a.budget,packet_directory=None)
    cfg['train_params'].update(epoches=20,batch_size=2)
    cfg['optimizer'].update(lr=5e-5,lr_warmup_epochs=1)
    cfg['lr_scheduler'].update(step_size=[14,18],gamma=.1)
    if a.arm=='pyramid':
        cfg['model']['core_method']='e3dsnn_snapshot_pyramid'
        cfg['optimizer'].update(new_parameter_prefix='joint_receiver.snapshot_pyramid.',new_parameter_lr=1e-3)
    root=Path(a.root);root.mkdir(parents=True,exist_ok=False)
    inputs=[source,Path(a.config),Path(a.manifest)]
    if a.mode=='admission':
        assert a.arm=='pyramid'
        state=torch.load(str(source),map_location='cpu')
        args=cfg['model']['args']
        depth=int(round((args['lidar_range'][5]-args['lidar_range'][2])/args['voxel_size'][2]))+1
        for _ in range(3):depth=(depth+1)//2
        depth=(depth-3)//2+1
        module=PyramidCurrent(depth)
        additions={'joint_receiver.snapshot_pyramid.'+k:v for k,v in module.state_dict().items()}
        assert not set(state).intersection(additions);state.update(additions)
        torch.save(state,str(root/'initial.pth'))
        config=root/'audit.yaml';config.write_text(yaml.safe_dump(cfg));inputs.append(config)
        jobs=[job('unit','opencood.tools.test_e3dsnn_pyramid_current',[],3600),
              job('audit','opencood.tools.audit_cp_pyramid',['--config',config,
              '--checkpoint',root/'initial.pth','--parent',source,'--output',root/'audit.json'],
              3600,[root/'audit.json'])]
    else:
        admission=Path(a.admission_root)
        ap=json.loads((admission/'protocol.json').read_text());ast=json.loads((admission/'status.json').read_text())
        report=json.loads((admission/'audit.json').read_text())
        assert ast['state']=='complete' and ast['protocol_sha256']==sha(admission/'protocol.json')
        assert ap['source_sha256']==F0_SHA and report['passed'] and report['budget']==a.budget
        for path,digest in ap['input_hashes'].items():assert sha(Path(path))==digest
        for rel in MODEL_FILES:
            assert subprocess.check_output(['git','show',ap['code_commit']+':'+rel],cwd=str(REPO))==(REPO/rel).read_bytes()
        inputs += [admission/'protocol.json',admission/'status.json',admission/'audit.json']
        initial=admission/'initial.pth' if a.arm=='pyramid' else source
        shutil.copyfile(str(initial),str(root/'initial.pth'))
        config=root/'train.yaml';config.write_text(yaml.safe_dump(cfg));inputs.append(config)
        jobs=[]
        if a.arm=='control':
            jobs.append(job('control_audit','opencood.tools.audit_cp_frontend_budget',
                ['--config',config,'--checkpoint',root/'initial.pth','--output',root/'audit.json'],3600,[root/'audit.json']))
        jobs.append(job('train','opencood.tools.train_e3dsnn_single',
            ['--mode','train','--config',config,'--checkpoint',root/'initial.pth',
             '--output',root/'train','--epochs',20,'--workers',2,'--seed',20260929,
             '--selection-iou',.7,'--retain-initial-best'],604800,
            [root/'train/status.json',root/'train/best_ap70.pth']))
        for b in (a.budget,0,8192 if a.budget==2304 else 2304):
            c=copy.deepcopy(cfg);c['model']['args']['snapshot'].update(budget_bytes=b,
                packet_directory=str(root/('b'+str(b))/'packets'))
            path=root/('b'+str(b)+'.yaml');path.write_text(yaml.safe_dump(c));inputs.append(path)
            jobs+=eval_jobs(root,b,path,root/'train/best_ap70.pth',a.reference)
        reference=Path(a.reference);inputs += [reference/'common_gt_manifest.json',reference/'common_val.json']
        for row in json.loads((reference/'common_gt_manifest.json').read_text())['frames']:
            p=reference/'common_gt'/(row['frame_id']+'.npy');assert sha(p)==row['sha256'];inputs.append(p)
    inputs += [root/'initial.pth']+[Path(cfg[k]) for k in ('root_dir','validate_dir')]
    for folder in a.dependency_paths:
        inputs += [p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inputs += list((REPO/'opencood/utils').glob('box_overlaps*.so'))
    protocol=dict(repo=str(REPO),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        input_hashes={str(p):sha(p) for p in inputs},jobs=jobs,dependency_paths=a.dependency_paths,
        source_sha256=F0_SHA,arm=a.arm,primary_budget=a.budget,mode=a.mode,seed=20260929,
        hypothesis='Does spatial reconstruction and multiscale current interaction improve AP70 at matched training and budget?',
        training='20 epochs, full frontend/backend; old lr5e-5 new lr1e-3; linear LR warmup epoch1; decay after14/18; frozen BN; Ldet+.5Lego; dropout .1',
        selection='Full native validation AP70 at the registered budget including epoch0; fixed best common CPU GT at 0/2304/8192B',
        exit_rule='No extensions or post-hoc selection; admission updates discarded; any failed job stops queue',
        scope='Receiver-only new computation; IF8 32-channel actual wire and membrane Count4 preserved',
        boundary='Single-seed architecture exploration, not a causal attention ablation or energy claim')
    dump(root/'protocol.json',protocol)
    print(json.dumps(dict(root=str(root),jobs=len(jobs),arm=a.arm,budget=a.budget,initial_sha256=sha(root/'initial.pth'))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','config','checkpoint','data-root','manifest'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=('admission','train'),required=True)
    p.add_argument('--arm',choices=('control','pyramid'),default='pyramid')
    p.add_argument('--budget',type=int,choices=(2304,8192),required=True)
    p.add_argument('--admission-root');p.add_argument('--reference')
    p.add_argument('--dependency-paths',nargs='*',default=[])
    a=p.parse_args()
    if a.mode=='train' and not (a.admission_root and a.reference):p.error('train needs admission and reference')
    prepare(a)
