"""Verified epoch-boundary continuation of completed spatial recovery runs."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_completed_spatial_resume(directory,contract,additional_epochs_max=12):
    if not isinstance(additional_epochs_max,int) or isinstance(additional_epochs_max,bool) or additional_epochs_max<1:
        raise ValueError('Additional epoch budget must be a positive integer')
    root=Path(directory)
    def read(name):return json.loads((root/name).read_text())
    status=read('status.json')
    if status['status']!='COMPLETED':raise ValueError('Source run must have completed; never read live latest')
    report=read('report.json')
    if report['stop_reason']!='budget_not_convergence':raise ValueError('Only a budget-limited run may continue automatically')
    saved=torch.load(str(root/'latest.pth'),map_location='cpu')
    original={k:v for k,v in saved['contract'].items() if k!='resume'}
    if original!=contract:raise ValueError('Resume architecture, objective or data contract mismatch')
    history=read('history.json')
    if not history or history!=saved['history']:raise ValueError('Checkpoint and history mismatch')
    epoch=history[-1]['epoch']
    if [h['epoch'] for h in history]!=list(range(1,epoch+1)) or status['epochs']!=epoch or report['epochs']!=epoch:
        raise ValueError('Resume requires a completed contiguous epoch boundary')
    final=torch.load(str(root/('epoch_%03d'%epoch)/'stage.pth'),map_location='cpu')
    if final['epoch']!=epoch or final['contract']!=saved['contract']:
        raise ValueError('Final stage checkpoint identity mismatch')
    if final['stage'].keys()!=saved['stage'].keys() or any(not torch.equal(v,saved['stage'][k]) for k,v in final['stage'].items()):
        raise ValueError('Latest weights do not match completed final epoch')
    best=read('best.json')
    if best!=report['best'] or sha256(best['checkpoint'])!=best['sha256']:
        raise ValueError('Best checkpoint identity mismatch')
    # The significant-best threshold differs from the absolute best AP. Recover
    # the last threshold-crossing epoch, not the largest tiny improvement.
    significant=saved.get('significant')
    if significant is None:
        resets=[h for h in history if h['stale']==0]
        significant=(resets[-1]['validation']['results']['full']['ap70'] if resets else
                     read('epoch_000/report.json')['results']['full']['ap70'])
    if saved['optimizer']['param_groups'][0]['lr']!=history[-1]['lr']:
        raise ValueError('Optimizer learning rate differs from completed history')
    for key in ('optimizer','scheduler','stage'):assert key in saved
    return dict(saved=saved,best=best,epoch=epoch,significant=significant,stale=history[-1]['stale'],
        provenance=dict(source_run=str(root),latest_sha256=sha256(root/'latest.pth'),
            completed_epoch=epoch,additional_epochs_max=additional_epochs_max,
            policy='Restore weights, AdamW moments, scheduler, significant best and global epoch. No learning-rate reset. New output; source unchanged.'))
