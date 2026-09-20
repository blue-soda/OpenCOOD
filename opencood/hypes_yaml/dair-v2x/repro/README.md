# DAIR-V2X-C Reproduction Entry Points

This directory includes current entry points and historical experimental configs.
Do not assume every YAML is a validated reproduction route. The current workspace
runbook is `C:\Workspace\opencood\agent-doc\reproduction.md` (updated 2026-09-20).

## Shared Reader

All asynchronous DAIR Stage2-style configs use:

```yaml
fusion:
  core_method: CoDynTrustDAIRIrregularFlowDataset
```

The reader lives at:

```text
opencood/data_utils/datasets/codyntrust_dair_irregular_flow_dataset.py
```

It was migrated from the reproduced CoDynTrust fork:

```text
git@github.com:blue-soda/CoDynTrust.git
branch: repro/dair-cobevflow-codyntrust-20260912
commit: ba359e2
```

## Configs

| Config | Purpose |
| --- | --- |
| `dair_stage1_codyntrust_single_wide.yaml` | checkpoint-compatible wide single Stage1, using the dedicated CoDynTrust shrink/BN structure |
| `dair_cobevflow_stage2_wide.yaml` | CoBEVFlow baseline Stage2 using `point_pillar_codyntrust_cobevflow_baseline` |
| `dair_codyntrust_stage2.yaml` | CoDynTrust Stage2 using `point_pillar_codyntrust` |

`dair_stage1_wide_single.yaml` uses the generic `point_pillar` model and must not
replace the checkpoint-compatible initializer above. CoDynTrust Stage2 is a
migrated entry point, not a claim that the historical reference-fork AP has been
retrained with this exact config. Historical and current voxel heights differ.

## Reference Results

Current high-value DAIR checkpoints and AP values are tracked outside git in:

```text
C:\Workspace\opencood\agent-doc\reproduction.md
C:\Workspace\opencood\agent-doc\milestone.md
```

Checkpoint files are intentionally not committed.
