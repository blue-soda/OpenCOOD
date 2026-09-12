# DAIR-V2X-C Reproduction Entry Points

This directory contains the clean DAIR reproduction entry points kept in this
OpenCOOD fork. Older experimental DAIR CoBEVFlow configs were removed to avoid
accidental use of incomplete or failed pipelines.

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
| `dair_stage1_wide_single.yaml` | wide single-agent Stage1 initializer |
| `dair_cobevflow_stage2_wide.yaml` | CoBEVFlow baseline Stage2 using `point_pillar_codyntrust_cobevflow_baseline` |
| `dair_codyntrust_stage2.yaml` | CoDynTrust Stage2 using `point_pillar_codyntrust` |

## Reference Results

Current high-value DAIR checkpoints and AP values are tracked outside git in:

```text
C:\Workspace\opencood\checkpoints\codyntrust_repro\README.md
```

Checkpoint files are intentionally not committed.
