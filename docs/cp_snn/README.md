# 协同感知 SNN：服务器实验入口

**后续进展：** 已移植 E-3DSNN 3D/BEV 主干并接入单阶段车端检测头，完成真实单帧过拟合和批量解码验证。
正式训练与定时跟进入口见 [单端基线训练记录](TRAINING.md)。下文保留前一阶段的数据/算子准备证据。
已有 PointPillars 的新协议评估、同结构 ANN 启动和证据索引见 [研究关键记录](RESEARCH_LOG.md)，
该记录同步至用户指定的 `agent-doc/snn-plans/cp-snn-plan`。
当前优先完成单车基线；逐帧预测归档、CPU AP 复算和推理计时说明见 [单车推理核验](INFERENCE.md)。

工作方式：本地 `C:\Workspace\OpenCOOD\OpenCOOD` 的 **main** 修改、提交并 push；
服务器 `mindspore-184:/data0/chen/gzc/workspace/OpenCOOD` 的 **main** pull 后运行。
已有研究代码以 fast-forward 合入 main，保留提交历史，不重置既有分支或未提交文件。

## 当前决策

从服务器已有 **DAIR-V2X-C** 开始，V2V4Real 留作车车协同验证。无需再下载 KITTI。
这是在协同数据上从头训练 E-3DSNN 方法的适配研究，不能称为复现论文 KITTI AP。
本地权重审计中，公开 kitti.pth 与当前官方 SNN 配置仅 59/341 个 state-dict 项兼容；
标准 ANN VoxelRCNN 结构可匹配 257/257 项（需 spconv 布局转换）。
因此暂不把该权重作为 SNN 预训练起点；这并不证明权重对所有用途都不可用。
审计原件位于 `C:\Workspace\E-3DSNN\local_setup\CHECKPOINT_AUDIT.md`。

## 首轮验证的边界

`opencood.tools.cp_snn_readiness` 使用现有 CoDynTrust DAIR reader：
每次查询仅车端当前帧和配对路端帧，p=0、k=1、无额外延迟、无历史、无姿态噪声、无增强。
从 train/val 各等距选择 3 个索引，检查点云、GT、两端 collate 和 SE(3) 逆变换；
遇到坏样本直接报错并保存部分报告，不悄悄换帧。
reader 的 timestamp 实为帧号，因此另从两侧 data_info.json 记录真实传感器时间戳。
p=0 只表示不人为添加延迟，不保证传感器物理同步。

原配置的 pillar 网格为 [0.4,0.4,4]，范围 [-100.8,-40,-3,100.8,40,1]。
独立的 GPU 算子探针改用 [0.4,0.4,0.1] 三维体素，保留相同空间范围；
执行 mean VFE → SubMConv3d(4,16) → BN → Count4 → backward。
Count4 是 floor(clamp(x,0,4)+0.5)，使用 (0,4) 区间的替代梯度，
参照官方 [E-3DSNN 源码](https://github.com/bollossom/E-3DSNN/blob/dbe5d1731d3204850bd591a18f7255ca2d1b6712/det/pcdet/models/backbones_3d/spconv_backbone_spike.py) 独立实现。
两端点云以独立 batch 索引进入同一个算子；探针直接体素化原始点云，未套用 reader 的去自车点预处理。
随机初始化、训练模式 BN、16 通道的小探针仅检查环境；它不是 E-3DSNN 主干、消息融合或检测器。
不据此推断 AP、训练后的稀疏率、能耗、延迟或通信收益。
计数仍使用 float32 存储；0..4 共五个取值，独立定长编码至少 3 bit，不能声称天然 2 bit。

## 运行

复用已存在的 opencood 环境，不修改共享依赖：

```bash
cd /data0/chen/gzc/workspace/OpenCOOD
git pull --ff-only origin main
export PYTHONPATH="$PWD"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 运行前检查 nvidia-smi，选择空闲卡。
CUDA_VISIBLE_DEVICES=1 /data0/chen/miniconda3/envs/opencood/bin/python \
  -m opencood.tools.cp_snn_readiness --samples 3 \
  --output /data0/chen/gzc/workspace/diagnostics/cp_snn_20260929/readiness.json
```

报告包含代码提交、工作区状态、配置和划分 SHA256、软件版本、索引、时间戳、
体素数、计数分布、梯度和 GPU 内存峰值。索引总量不等于全量样本均已验证。
诊断配置仍属于现有 ANN reader，不能直接用它启动“SNN 训练”。

## 2026-09-29 实测结果

代码 `388035953c9bb6b1e19c50310c18c9452204d0d0` 已经本地 push、服务器 pull 后执行。
[原始 JSON 报告](results/readiness_20260929.json) 状态为 passed，进程退出码为 0。
服务器日志及运行前未跟踪文件备份保存在
`/data0/chen/gzc/workspace/diagnostics/cp_snn_20260929/`。

- 环境：Python 3.7.11、PyTorch 1.10.0+cu113、spconv 2.3.6、NumPy 1.21.6、RTX 3090（GPU 1）。
- 索引：train 4,811，val 1,789；本次只抽查各 3 帧，不是全量清洗或全量评估。
- 六个样本的真实两端点云、GT、collate、SE(3) 检查与稀疏 CUDA 前向/反向均通过。
- 两端合计三维活动体素 27,922–38,364；每体素输出 16 通道，值位于整数集合 {0,1,2,3,4}。
- Count4 边界与替代梯度检查通过，点特征和稀疏卷积权重均得到有限且非零梯度。
- 两端原始时间戳差值均非零，具体值留在报告；不能用 reader 的 time_diff=0 证明物理同步。
- torch 峰值 allocated 为 27,433,472 bytes，只包含这个小探针的 PyTorch 分配统计，不能外推完整主干显存。

这确认的是服务器的数据与基础算子可用。尚未实现完整 E-3DSNN 检测器、训练、通信编码或协同融合；
没有新的 AP、实际消息 bytes 或节能结论。已有 SNN 预训练权重不作为本次探针的依赖。

## 接下来的模型实验

1. 移植并核对 E-3DSNN 三维计数主干，建立 DAIR 单端 SNN 检测链，先单 batch 过拟合，再训练单端基线。
2. 固定数据范围、检测头和训练预算，设置 ANN 以及 INT2/4/8 对照；区分主干变化和量化收益。
3. 在同一 SNN 链上导出空间坐标与计数；车路共享编码器，先做真实 SE(3) 对齐、再体素化及冲突聚合，再进入 BEV/检测头。
4. 明确测量坐标、计数、时间戳、报头和打包后的实际 bytes；加入协同训练后再比较 AP 与通信量。
5. 最后研究跨消息到达的膜状态更新。E-3DSNN 当前 Count4 本身没有跨包膜状态；虚拟脉冲步、物理时间与发送者 ID 必须分开。

不启动长时间训练，直到完整检测链、单 batch loss/梯度以及对照配置通过核验。
