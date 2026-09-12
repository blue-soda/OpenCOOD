# Agent Quick Start

本仓库当前用于 DAIR-V2X-C / V2V4Real 上的协同感知同步与异步复现实验，
以及 ECTRA 方法开发。完整工作区说明在：

```text
C:\Workspace\opencood\agent-doc\workspace_onboarding.md
```

服务器：

```bash
ssh chen@mindspore-184
cd /data0/chen/gzc/workspace/OpenCOOD
export PYTHONPATH=$PWD
PY=/data0/chen/miniconda3/envs/opencood/bin/python
```

当前可信配置：

```text
opencood/hypes_yaml/dair-v2x/repro/dair_stage1_wide_single.yaml
opencood/hypes_yaml/dair-v2x/repro/dair_cobevflow_stage2_wide.yaml
opencood/hypes_yaml/dair-v2x/repro/dair_codyntrust_stage2.yaml
opencood/hypes_yaml/dair-v2x/npj/dair_ectra_stage2_roi.yaml
opencood/hypes_yaml/v2v4real/npj/v2v4real_cobevflow_stage1_sync_where2comm_trafcost_aligned.yaml
```

DAIR 异步链路统一 reader：

```text
opencood/data_utils/datasets/codyntrust_dair_irregular_flow_dataset.py
fusion.core_method: CoDynTrustDAIRIrregularFlowDataset
```

高价值 checkpoint 和论文/复现 AP 对照见：

```text
C:\Workspace\opencood\checkpoints\codyntrust_repro\README.md
C:\Workspace\opencood\agent-doc\workspace_onboarding.md
```

不要恢复早期 `CoBEVFlowDAIRIrregularDataset`、旧 `dair_cobevflow_*`
试验配置、旧 V2V4Real `max_multiscale` 配置或旧 motion-prediction 配置；
它们已因链路不完整或 AP 异常被清理。
