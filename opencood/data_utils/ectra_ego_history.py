"""Causal DAIR ego references without changing collaborator delay sampling."""

from bisect import bisect_right
from collections import defaultdict
from pathlib import PurePosixPath


class CausalEgoHistoryIndex:
    def __init__(self, vehicle_info, max_age_ms=100):
        self.max_age_us = int(max_age_ms * 1000)
        if self.max_age_us < 0:
            raise ValueError('max_age_ms must be nonnegative')
        self.by_id = {}
        self.batches = defaultdict(list)
        for info in vehicle_info:
            fid = PurePosixPath(info['pointcloud_path']).stem
            self.by_id[fid] = info
            self.batches[str(info['batch_id'])].append(
                (int(info['pointcloud_timestamp']), fid))
        for rows in self.batches.values():
            rows.sort()

    def select(self, current_vehicle_id, infrastructure_timestamp):
        current = self.by_id[current_vehicle_id]
        now = int(current['pointcloud_timestamp'])
        target = int(infrastructure_timestamp)
        rows = self.batches[str(current['batch_id'])]
        # No observation may be later than either the collaborator or current ego.
        idx = bisect_right(rows, (min(now, target), '\uffff')) - 1
        if idx < 0:
            return None, None
        timestamp, fid = rows[idx]
        age_us = target - timestamp
        if age_us > self.max_age_us:
            return None, age_us / 1000.0
        return fid, age_us / 1000.0
