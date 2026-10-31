"""Register fresh paired event-quota runs; preserve predecessor byte-cap evidence."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import yaml
from opencood.tools.prepare_cp_budget_round import REPO, job, eval_jobs, dump
from opencood.tools.run_registered_research import sha


def prepare(a):
    old=Path(a.predecessor);root=Path(a.root)
    previous=json.loads((old/'protocol.json').read_text())
    cfg=yaml.safe_load((old/'train.yaml').read_text())
    assert cfg['model']['core_method'] in ('e3dsnn_snapshot_frontend','e3dsnn_snapshot_pyramid')
    assert cfg['train_params']['epoches']==20 and cfg['train_params']['batch_size']==2
    budget=cfg['model']['args']['snapshot']['budget_bytes'];assert budget in (2304,8192)
    root.mkdir(parents=True,exist_ok=False)
    cfg['model']['args']['snapshot'].update(selection='event_quota',
        event_quotas={2304:2500,8192:19000},packet_directory=None)
    path=root/'train.yaml';path.write_text(yaml.safe_dump(cfg))
    shutil.copyfile(str(old/'initial.pth'),str(root/'initial.pth'))
    assert sha(old/'initial.pth')==sha(root/'initial.pth')
    jobs=[job('audit','opencood.tools.audit_snapshot_event_quota',
        ['--config',path,'--checkpoint',root/'initial.pth','--output',root/'audit.json'],900,[root/'audit.json']),
        job('train','opencood.tools.train_e3dsnn_single',
        ['--mode','train','--config',path,'--checkpoint',root/'initial.pth','--output',root/'train',
         '--epochs',20,'--workers',2,'--seed',20260929,'--selection-iou','0.7',
         '--retain-initial-best','--defer-initial-validation'],345600,
        [root/'train/status.json',root/'train/best_ap70.pth'])]
    jobs[-1]['progress_watch']=dict(directory=str(root/'train'),timeout_seconds=900)
    for label,quota in ((0,0),(2304,2500),(8192,19000)):
        c=copy.deepcopy(cfg);c['model']['args']['snapshot'].update(budget_bytes=label,
            packet_directory=str(root/('b'+str(label))/'packets'))
        p=root/('b'+str(label)+'.yaml');p.write_text(yaml.safe_dump(c))
        terminal=eval_jobs(root,label,p,root/'train/best_ap70.pth',a.reference)
        terminal[-1]['command']+=['--event-quota',str(quota)]
        jobs+=terminal
    inputs=[root/'initial.pth',old/'protocol.json',old/'train.yaml']+list(root.glob('*.yaml'))
    inputs += [Path(cfg[k]) for k in ('root_dir','validate_dir','data_manifest')]
    inputs += [Path(a.reference)/'common_gt_manifest.json',Path(a.reference)/'common_val.json']
    inputs += list((REPO/'opencood').rglob('*.py'))+list((REPO/'opencood').rglob('*.so'))
    protocol=dict(repo=str(REPO),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(REPO)).decode().strip(),
        dependency_paths=previous.get('dependency_paths',[]),input_hashes={str(p):sha(p) for p in inputs},jobs=jobs,
        study='R-044 event-quota paired architecture study',predecessor=str(old),predecessor_protocol_sha256=sha(old/'protocol.json'),
        primary_event_quota=2500 if budget==2304 else 19000,nominal_budget_label=budget,
        hard_compressed_byte_cap=False,seed=20260929,initial_sha256=sha(root/'initial.pth'),
        quota_rationale='Fixed before new training: rounded historical training event means about 2450/18700; not selected on AP',
        selection='Full validation AP70 at fixed primary event quota; include epoch0; terminal immutable prediction replay on common GT',
        training='Fresh original F0 or matched zero-pyramid initial tensors, 20 epochs, batch2, old lr5e-5/new1e-3, warmup1, decay14/18, frozen BN, Ldet+.5Lego, message dropout .1',
        communication='No encode/decode/byte statistics in training or routine validation; terminal evaluation encodes once and decodes once per message, reports measured mean/P95/max',
        migration='New scientific protocol, not an exact continuation. Preserve predecessor checkpoints and progress; no mixing old byte-cap AP with event-quota AP',
        exit_rule='No extension or post-hoc quota selection; report all four arms and all three terminal quotas',
        scope='Preserve SNN encoder, IF crossing-time order, sparse count message, 3D alignment, local membrane, Count4 and ANN detector')
    dump(root/'protocol.json',protocol)
    print(json.dumps(dict(root=str(root),protocol_sha256=sha(root/'protocol.json'),primary_event_quota=protocol['primary_event_quota'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('predecessor','root','reference'):p.add_argument('--'+name,required=True)
    prepare(p.parse_args())
