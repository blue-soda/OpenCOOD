"""Replay the final matched gates on their development and transfer cohorts."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opencood.tools.check_int_deep_block_adapt import TRAINED, make_trainable, tensor_digest
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import checked_plan
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, build
from opencood.tools.run_int_ego import write_json
from opencood.tools.train_int_deep_block_adapt import (
    EPOCHS, TRAIN_ARMS, compile_trace, evaluate,
)


REPLICATES = 2000
BOOTSTRAP_SEED = 303
CONTRASTS = (
    ('local_trained','frozen_local'),
    ('local_trained','whole_trained'),
    ('local_trained','p5'),
    ('whole_trained','frozen_local'),
)


def read(path):
    return json.loads(Path(path).read_text())


def average_precision(scores, true_positive, count):
    if count == 0:
        raise ValueError('Scene bootstrap sample had no ground truths')
    order = np.argsort(-scores)
    tp = np.cumsum(true_positive[order], dtype=np.int64)
    recall = np.r_[0., tp / count, 1.]
    precision = np.r_[0., tp / np.arange(1,len(tp)+1), 0.]
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.dot(np.diff(recall),envelope[1:]))


def bootstrap(directory, metrics, seed):
    records = read(directory/'ap_records.json')
    index = read(directory/'scene_slices.json')
    assert len({part['scene'] for part in index}) == len(index)
    parts = {}
    for arm in records:
        score = np.asarray(records[arm]['0.7']['score'])
        tp = np.asarray(records[arm]['0.7']['tp'], dtype=np.int64)
        bounds = [part['arms'][arm]['0.7'] for part in index]
        assert bounds[0][0] == 0 and bounds[-1][1] == len(score)
        assert all(a[1] == b[0] for a,b in zip(bounds,bounds[1:]))
        per_scene = [(score[lo:hi],tp[lo:hi],gt) for lo,hi,gt in bounds]
        assert sum(p[2] for p in per_scene) == records[arm]['0.7']['gt']
        parts[arm] = per_scene

    def aps(sample):
        results = {}
        for arm, choices in parts.items():
            chunks = [choices[i] for i in sample]
            results[arm] = average_precision(np.concatenate([p[0] for p in chunks]),
                                            np.concatenate([p[1] for p in chunks]),
                                            sum(p[2] for p in chunks))
        return results

    baseline = aps(range(len(index)))
    assert all(abs(v-metrics[arm]['ap70'])<1e-12 for arm,v in baseline.items())
    samples = np.random.RandomState(seed).randint(len(index),size=(REPLICATES,len(index)))
    gt = np.asarray([record[2] for record in parts['p5']])
    assert all([record[2] for record in values] == gt.tolist() for values in parts.values())
    results = [aps(row) for row in samples if gt[row].sum()>0]
    distinctions = {}
    for left, right in CONTRASTS:
        gains = np.array([100*(row[left]-row[right]) for row in results])
        distinctions[left+'_minus_'+right] = dict(estimate_pp=100*(baseline[left]-baseline[right]),
            percentile95_pp=np.percentile(gains,[2.5,97.5]).tolist(), positive_fraction=float((gains>0).mean()))
    payload = dict(scenes=[row['scene'] for row in index], seed=seed, replicates=REPLICATES, valid_replicates=len(results),
                   sampling='Paired within cohort, resample entire scenes with replacement',
                   differences=distinctions)
    write_json(directory/'paired_bootstrap.json',payload)
    return payload


def load_models(parent, checkpoint, saved_identity):
    serialized_identity = json.loads(json.dumps(checkpoint['identity']))
    assert serialized_identity == saved_identity and checkpoint['epoch'] == EPOCHS
    frozen = tensor_digest(parent.state_dict(),TRAINED)
    assert saved_identity['frozen_digest']==frozen
    models={}
    for arm in TRAIN_ARMS:
        model, _ = make_trainable(parent)
        weights = checkpoint['weights'][arm]
        assert set(weights)==set(TRAINED)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name in weights:
                    p.copy_(weights[name])
        assert tensor_digest(model.state_dict(),TRAINED)==frozen
        models[arm]=model
    return models


def run(args,out):
    primary,transfer=checked_plan(args.primary_plan),checked_plan(args.transfer_plan)
    train=checked_plan(args.train_plan)
    assert primary['split']==transfer['split']=='val' and train['split']=='train'
    scene_sets=[{s[0] for s in plan['scenes']} for plan in (train,primary,transfer)]
    assert all(a.isdisjoint(b) for i,a in enumerate(scene_sets) for b in scene_sets[i+1:])
    previous=Path(args.final_run)
    previous_report=read(previous/'report.json')
    assert previous_report['epoch']==EPOCHS and previous_report['full_budget'] and previous_report['status']=='COMPLETED'
    saved_identity=read(previous/'contract.json')['identity']
    assert saved_identity['source_sha256']==SOURCE_SHA
    assert saved_identity['plans']=={'train':digest(Path(args.train_plan)),'val':digest(Path(args.primary_plan))}
    latest=read(previous/'latest_checkpoint.json')
    cp=previous/('after_epoch_%02d.pth' % EPOCHS)
    assert latest['epoch']==EPOCHS and latest['sha256']==digest(cp)
    trace_path=previous/'fixed_traces.json'
    assert digest(trace_path)==saved_identity['trace_sha256']
    primary_trace=read(trace_path)['val']
    restart=torch.load(cp,map_location='cpu')
    args.kind='geometry'
    source,cfg,rows,_=build(args)
    source.period,source.max_age=5,.55
    before=model_digest(source)
    models=load_models(source,restart,saved_identity)
    origin=Path(args.transfer_baseline)
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=SOURCE_SHA, train_identity=saved_identity, checkpoint_sha256=latest['sha256'],
        primary_plan_sha256=digest(Path(args.primary_plan)),transfer_plan_sha256=digest(Path(args.transfer_plan)),
        prior_primary_ap_sha256=digest(previous/'eval_08'/'ap_records.json'),
        prior_transfer_ap_sha256=digest(origin/'ap_records.json'),torch=torch.__version__,
        device=torch.cuda.get_device_name(), bootstrap_replicates=REPLICATES,
        note='Descriptive, separate per-cohort paired scene bootstrap. Transfer was used for scorer screening, not gate training or endpoint choice. Historical DAIR validation used in source work.'))
    write_json(out/'status.json',dict(status='RUNNING',phase='transfer_fixed_trace'))
    with torch.no_grad():
        transfer_trace=compile_trace(source,cfg,rows,transfer)
    write_json(out/'transfer_traces.json',transfer_trace)
    results={}
    for i,(name,plan,traces) in enumerate((('primary',primary,primary_trace),('transfer',transfer,transfer_trace))):
        path=out/name
        path.mkdir()
        write_json(out/'status.json',dict(status='RUNNING',phase=name))
        report=evaluate(EPOCHS,source,models,cfg,rows,plan,traces,path,include_scenes=True)
        scores=read(path/'eval_08'/'ap_records.json')
        if name=='primary':
            archived=read(previous/'eval_08'/'ap_records.json')
            assert scores==archived, 'Primary final score replay mismatched'
        else:
            archived=read(origin/'ap_records.json')
            for new,old in [('p5','p5'),('frozen_local','residual')]:
                assert scores[new]==archived[old], (new, 'Archived transfer baseline mismatched')
        results[name]=dict(report=report,bootstrap=bootstrap(path/'eval_08',report['metrics'],BOOTSTRAP_SEED+i))
    assert model_digest(source)==before
    write_json(out/'report.json',dict(status='COMPLETED',both_prior_score_replays_exact=True,cohorts=results,
        parameters=sum(p.numel() for p in source.parameters()),checkpoint_sha256=latest['sha256']))
    write_json(out/'status.json',dict(status='COMPLETED',primary=results['primary']['report']['scans'],
        transfer=results['transfer']['report']['scans']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','checkpoint','trained','readout','train-plan','primary-plan','transfer-plan',
                'final-run','transfer-baseline','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='RUNNING',phase='inputs'))
    try:
        run(args,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
