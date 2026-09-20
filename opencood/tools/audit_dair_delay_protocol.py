"""Replay the actual DAIR sampler without point-cloud I/O or protocol changes."""

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(str(out))
    cfg = load_yaml(args.config)
    dataset = build_dataset(cfg, visualize=False, train=args.split == 'train')
    if type(dataset).__name__ != 'CoDynTrustDAIRIrregularFlowDataset':
        raise ValueError('This audit targets the current DAIR reader')
    if dataset.only_async:
        raise ValueError('This audit requires the binomial branch')
    module = importlib.import_module(type(dataset).__module__)
    # All changes below are process-local I/O stubs, not modified sampling logic.
    module.pcd_utils.read_pcd = lambda *a, **kw: (np.zeros((0, 4)), None)
    module.load_json = lambda *a, **kw: []
    dataset.get_vehicle_trans = lambda *a: [0.] * 6
    dataset.get_inf_trans = lambda *a: [0.] * 6
    original_bernoulli = module.stats.bernoulli
    draws = []

    class TracedBernoulli:
        def __init__(self, p):
            self.distribution = original_bernoulli(p)

        def rvs(self, size):
            value = self.distribution.rvs(size)
            draws.append(int(np.sum(value)))
            return value

    module.stats.bernoulli = TracedBernoulli
    strict_original = dataset.strict_data
    dataset.strict_data = True
    strict_count = sum(dataset.is_valid_id(fid) for fid in dataset.data_split)
    dataset.strict_data = strict_original
    random.seed(args.seed)
    np.random.seed(args.seed)
    counts, requested_hist, actual_hist, latest_hist = Counter(), Counter(), Counter(), Counter()
    examples, errors = [], []
    for idx in range(len(dataset)):
        draws.clear()
        try:
            data = dataset.retrieve_base_data(idx)
        except Exception as error:
            errors.append(dict(index=idx, error=repr(error)))
            continue
        counts['samples'] += 1
        previous = data[1]['curr']['frame_id']
        current = int(previous)
        assert len(draws) == dataset.k
        sample_changed = False
        for j, requested in enumerate(draws):
            frame = data[1]['past_k'][j]
            actual = -int(frame['sample_interval'])
            assert actual == int(previous) - int(frame['frame_id'])
            assert int(frame['time_diff']) == int(frame['frame_id']) - current
            requested_hist[requested] += 1
            actual_hist[actual] += 1
            counts['steps'] += 1
            if j == 0:
                latest_hist[actual] += 1
            if actual != requested:
                sample_changed = True
                counts['changed_steps'] += 1
                boundary = int(previous) - requested < int(dataset.inf_idx2info[previous]['batch_start_id'])
                reason = 'sequence_boundary' if boundary else 'missing_pair'
                counts[reason] += 1
                # Counterfactual nearest valid predecessor, not used for inference.
                nearest = 0
                if not boundary:
                    for gap in range(requested, -1, -1):
                        if module.id_to_str(int(previous) - gap) in dataset.inf_fid2veh_fid:
                            nearest = gap
                            break
                    if nearest != actual:
                        counts['would_change_if_loop_fixed'] += 1
                if len(examples) < 12:
                    examples.append(dict(index=idx, step=j, previous=previous,
                                         requested=requested, actual=actual, reason=reason,
                                         nearest_valid_gap=nearest))
            previous = frame['frame_id']
        counts['changed_samples'] += int(sample_changed)

    def stats(hist):
        total = sum(hist.values())
        return dict(histogram=dict(sorted(hist.items())),
                    mean=sum(k * v for k, v in hist.items()) / max(total, 1),
                    zero_fraction=hist[0] / max(total, 1))

    result = dict(config=args.config, split=args.split, seed=args.seed,
                  n=dataset.binomial_n, p=dataset.binomial_p, k=dataset.k,
                  strict_data=dataset.strict_data, input_split_count=len(dataset.data_split),
                  current_reader_count=len(dataset), strict_eligible_count=strict_count,
                  counts=dict(counts), requested_steps=stats(requested_hist),
                  actual_steps=stats(actual_hist), latest_delay=stats(latest_hist),
                  examples=examples, errors=errors,
                  limitations='Metadata only; includes bad/empty PCD indices. Sequential RNG replay, not exact DataLoader/evaluation draw order. No AP or actual millisecond claim.')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
