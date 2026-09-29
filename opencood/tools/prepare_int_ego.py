"""Freeze causal ego streams from existing DAIR train/validation scene splits."""
import argparse
import json
from pathlib import Path

from opencood.data_utils.int_ego_stream import build_stream_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = build_stream_manifest(args.data)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'sequence_manifest.json').write_text(json.dumps(manifest, indent=2))
    smoke = dict(manifest)
    # Preserve causal prefix, including unlabeled warmup, through 32 labels.
    for split in ['train', 'val']:
        labels, end = 0, 0
        for end, row in enumerate(manifest[split], 1):
            labels += int(row['supervised'])
            if labels >= 32 and end >= 64:
                break
        smoke[split] = manifest[split][:end]
    smoke['scope'] = 'SMOKE ONLY: causal prefix through at least 32 labels and 64 scans'
    (directory/'smoke_manifest.json').write_text(json.dumps(smoke, indent=2))
    print(json.dumps(manifest['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
