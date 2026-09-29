"""Epoch sampling: shuffle independent segments, never truncate their history."""
from collections import defaultdict
import random


def segment_order(rows, seed):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row['frame'] in seen:
            raise ValueError('Duplicate frame: '+row['frame'])
        seen.add(row['frame'])
        groups[row['segment']].append(dict(row))
    segments = list(groups.values())
    for group in segments:
        stamps = [r['timestamp_us'] for r in group]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Noncausal segment')
        for i, row in enumerate(group):
            row['reset_before'] = i == 0
    random.Random(seed).shuffle(segments)
    return [r for group in segments for r in group]
