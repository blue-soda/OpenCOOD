"""Decompose IF8 snapshot packet bytes into header, JSON context and payload.

The IF8 wire format (``opencood/utils/task_snapshot_packet.py``) is

    HEADER (25 B) + JSON context (uncompressed) + zlib(raw)

    raw = 4*N row coordinates + 1-bit channel mask + ceil(K*bits/8) counts

where N is the number of non-zero spatial rows and K the number of non-zero
counts. This tool reports the measured components so a byte difference between
two runs can be attributed instead of guessed.

Read-only: no model, no GPU, no decoding beyond the header.
"""
import argparse
import json
import struct
from collections import Counter
from pathlib import Path

HEADER = struct.Struct('<5sIIIII')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', required=True)
    parser.add_argument('--glob', default='*.bin')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.packets)
    rows, values, json_bytes, comp_bytes, total_bytes = [], [], [], [], []
    for path in sorted(folder.glob(args.glob)):
        data = path.read_bytes()
        if len(data) < HEADER.size:
            continue
        magic, msize, n, nnz, rsize, crc = HEADER.unpack(data[:HEADER.size])
        if magic != b'E3TS1':
            raise ValueError('Unexpected packet magic in %s' % path)
        rows.append(n)
        values.append(nnz)
        json_bytes.append(msize)
        payload = len(data) - HEADER.size - msize
        comp_bytes.append(payload)
        total_bytes.append(len(data))
    frames = max(1, len(total_bytes))
    def mean(xs):
        return sum(xs) / frames
    result = {
        'packets_dir': str(folder),
        'frames': len(total_bytes),
        'mean_total_bytes': mean(total_bytes),
        'mean_header_bytes': HEADER.size,
        'mean_json_bytes': mean(json_bytes),
        'mean_payload_compressed_bytes': mean(comp_bytes),
        'mean_rows_N': mean(rows),
        'mean_values_K': mean(values),
        'mean_values_per_row': mean(values) / max(1e-9, mean(rows)),
        'max_total_bytes': max(total_bytes) if total_bytes else 0,
        'p95_total_bytes': sorted(total_bytes)[int(0.95 * (len(total_bytes) - 1))] if total_bytes else 0,
        'json_share': mean(json_bytes) / max(1e-9, mean(total_bytes)),
        'payload_share': mean(comp_bytes) / max(1e-9, mean(total_bytes)),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
