"""Relocate a frozen INT manifest/config without changing its experimental rows."""
import argparse
import copy
import json
from pathlib import Path

import yaml

from opencood.tools.launch_int_memory import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    expected_manifest = '18886ad3a55df76621b61cea67e1d352203e8cdf4cd8c36fec36ff72fe27951a'
    expected_checkpoint = 'c5fdef2fcc2914a09bb527cee6723730e70850056e3115f3197122e69431efb9'
    if digest(a.manifest) != expected_manifest or digest(a.checkpoint) != expected_checkpoint:
        raise ValueError('Source differs from frozen v3 data or original P1 initialization')
    source = json.loads(Path(a.manifest).read_text())
    config = yaml.safe_load(Path(a.config).read_text())
    old_root, new_root = Path(config['data_dir']), Path(a.data_root).resolve()
    relocated = copy.deepcopy(source)
    report = dict(source_manifest_sha256=expected_manifest,
                  checkpoint_sha256=expected_checkpoint, source_config_sha256=digest(a.config),
                  old_data_root=str(old_root), new_data_root=str(new_root), coverage={})
    for split, counts in (('train', (8593, 4678)), ('val', (5794, 1738))):
        rows = relocated[split]
        if (len(rows), sum(r['supervised'] for r in rows)) != counts:
            raise ValueError('Frozen population mismatch')
        for before, after in zip(source[split], rows):
            for key in ('pcd', 'label'):
                if before[key] is not None:
                    target = new_root / Path(before[key]).relative_to(old_root)
                    if not target.is_file():
                        raise FileNotFoundError(str(target))
                    after[key] = str(target)
            restored = dict(after, pcd=before['pcd'], label=before['label'])
            if restored != before:
                raise ValueError('Frame metadata/order changed during relocation')
        if digest(new_root/(split+'.json')) != source['protocol']['hashes'][split+'.json']:
            raise ValueError('Dataset split file differs')
        report['coverage'][split] = dict(scans=counts[0], labels=counts[1])
    for key in ('data_dir', 'root_dir', 'validate_dir', 'test_dir'):
        config[key] = str(new_root / Path(config[key]).relative_to(old_root))
    # No sampler rebuild: excluded corrupt PCD and original segment boundaries stay frozen.
    destination = Path(a.output)
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path, config_path = destination/'sequence_manifest.json', destination/'config.yaml'
    manifest_path.write_text(json.dumps(relocated, indent=2)+'\n')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report.update(status='PATHS_ONLY_VERIFIED', manifest=str(manifest_path), config=str(config_path),
                  manifest_sha256=digest(manifest_path), config_sha256=digest(config_path))
    (destination/'relocation_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
