"""Paired resident-CPU codec benchmark on one frozen real feature tensor."""
import argparse
import json
import subprocess
import time
import types
from pathlib import Path
import numpy as np
from opencood.utils import spike_packet as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-commit', default='98d7f43')
    parser.add_argument('--repeats', type=int, default=40)
    args = parser.parse_args()
    folder = Path(args.audit_dir)
    features = np.load(str(folder / 'first_features.npz'))
    stored = (folder / 'first_packet.bin').read_bytes()
    _, _, metadata = current.decode(stored)
    reference = types.ModuleType('reference_codec')
    source = subprocess.check_output(['git', 'show', args.reference_commit + ':opencood/utils/spike_packet.py'])
    exec(compile(source, '<reference_codec>', 'exec'), reference.__dict__)
    coords, values = features['coords'], features['features']
    modules = {'reference': reference, 'optimized': current}
    for module in modules.values():
        assert module.encode(coords, values, metadata) == stored
        for _ in range(3):
            module.decode(module.encode(coords, values, metadata))
    timings = {name: {'encode': [], 'decode': []} for name in modules}
    for index in range(args.repeats):
        names = ['reference', 'optimized'] if index % 2 == 0 else ['optimized', 'reference']
        for name in names:
            module = modules[name]
            start = time.perf_counter()
            packet = module.encode(coords, values, metadata)
            timings[name]['encode'].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            restored = module.decode(packet)
            timings[name]['decode'].append((time.perf_counter() - start) * 1000)
            assert packet == stored
            np.testing.assert_array_equal(restored[1], values[np.any(values, axis=1)])
    report = {'passed': True, 'frame_id': metadata['frame_id'], 'packet_bytes': len(stored),
        'byte_identical_to_reference': True, 'reference_commit': args.reference_commit,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'repeats_per_method': args.repeats, 'alternating_order': True, 'warmup_iterations': 3,
        'timings_ms': {name: {op: {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95))}
                              for op, times in ops.items()} for name, ops in timings.items()},
        'scope': 'one frozen frame on shared CPU, resident arrays, excludes GPU transfers and transport'}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
