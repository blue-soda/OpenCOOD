"""Profile deterministic dataset reads without running a detector."""

import argparse
import cProfile
import faulthandler
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml


class ProfiledDataset(torch.utils.data.Dataset):
    """Keep each worker's stacks and timings separate while preserving samples."""

    def __init__(self, dataset, output, stack_interval):
        self.dataset = dataset
        self.output = output
        self.stack_interval = stack_interval

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prefix = self.output / ('worker_%d' % os.getpid())
        with Path(str(prefix) + '_stacks.log').open('a') as stacks:
            stacks.write('\nDATASET INDEX %d\n' % index)
            stacks.flush()
            start = time.perf_counter()
            profiler = cProfile.Profile()
            error = None
            faulthandler.dump_traceback_later(self.stack_interval, repeat=True,
                                            file=stacks)
            try:
                profiler.enable()
                return self.dataset[index]
            except Exception as exc:
                error = '%s: %s' % (type(exc).__name__, exc)
                raise
            finally:
                profiler.disable()
                faulthandler.cancel_dump_traceback_later()
                elapsed = time.perf_counter() - start
                profiler.dump_stats(str(prefix) + '_latest.prof')
                with Path(str(prefix) + '_samples.jsonl').open('a') as records:
                    records.write(json.dumps({'index': index, 'seconds': elapsed,
                                              'error': error}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--stack_interval', type=float, default=30)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    if (args.start < 0 or args.count <= 0 or args.stack_interval <= 0
            or args.num_workers < 0):
        parser.error('start/workers >= 0, count/stack_interval > 0 required')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'arguments.json').write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # DataLoader workers also use one torch CPU thread.
    torch.set_num_threads(1)
    dataset = build_dataset(load_yaml(args.hypes_yaml), visualize=False,
                            train=False)
    if args.num_workers:
        loader = torch.utils.data.DataLoader(
            ProfiledDataset(dataset, output, args.stack_interval), batch_size=1,
            sampler=range(args.start, min(args.start + args.count, len(dataset))),
            num_workers=args.num_workers, collate_fn=dataset.collate_batch_test,
            generator=torch.Generator().manual_seed(args.seed))
        last = time.perf_counter()
        with (output / 'received.jsonl').open('w') as records:
            for index, batch in enumerate(loader, args.start):
                now = time.perf_counter()
                row = {'index': index, 'wait_seconds': now - last,
                       'empty': batch is None}
                records.write(json.dumps(row) + '\n')
                records.flush()
                print(json.dumps(row), flush=True)
                del batch
                last = time.perf_counter()
        return
    profiler = cProfile.Profile()
    with (output / 'samples.jsonl').open('w') as records, \
            (output / 'slow_stacks.log').open('w') as stacks:
        for index in range(args.start, min(args.start + args.count, len(dataset))):
            print('reading dataset index %d' % index, flush=True)
            stacks.write('\nDATASET INDEX %d\n' % index)
            stacks.flush()
            start = time.perf_counter()
            error = None
            faulthandler.dump_traceback_later(args.stack_interval, repeat=True,
                                            file=stacks)
            try:
                profiler.enable()
                sample = dataset[index]
                empty = sample is None
                del sample
            except Exception as exc:
                error = '%s: %s' % (type(exc).__name__, exc)
                raise
            finally:
                profiler.disable()
                faulthandler.cancel_dump_traceback_later()
                record = {'index': index,
                          'seconds': time.perf_counter() - start,
                          'error': error}
                if error is None:
                    record['empty'] = empty
                records.write(json.dumps(record) + '\n')
                records.flush()
                profiler.dump_stats(str(output / 'dataset.prof'))
                print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
