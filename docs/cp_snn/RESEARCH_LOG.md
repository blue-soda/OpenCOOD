# 协同感知与 SNN：单端基线关键记录

日期：2026-09-29。此文件同步至用户指定的
`C:\Workspace\OpenCOOD\agent-doc\snn-plans\cp-snn-plan\04_单端基线与实验进展_20260929.md`。
代码在 `C:\Workspace\OpenCOOD\OpenCOOD` 的 main 修改、push，服务器 main pull。
本记录区分实测结果、进行中的训练和计划，不把阶段验证当作最终结论。

## 1. 数据与评估协议

- 数据：服务器已有 **DAIR-V2X-C**，即 DAIR-V2X 的车路协同部分。当前实验只输入车端当前帧，没有路端特征、历史帧或融合。
- 根目录：`/data0/chen/gzc/dataset/dair-v2x-c/extracted/cooperative-vehicle-infrastructure`。
- 原始 train/val：4,811 / 1,789；全量 PCD 审计后有效样本：**4,678 / 1,738**。
- 排除 133 / 51 个读取为空的点云，包含零字节和损坏文件；排除清单和源划分哈希均已归档。
- GT：车端本地标签；空间范围 [-100.8,-40,-3.5,100.8,40,1.5]。
- 指标：完整有效验证集的 **BEV AP@0.3/0.5/0.7**，不是 3D AP，也不是协同标签下 AP。
- 同一范围内，验证 GT 共 **22,971** 框。最终对照需核验 GT、样本和配置一致。

## 2. 当前 SNN 到底是什么

取自 E-3DSNN 官方 KITTI Voxel R-CNN 分支（源提交 dbe5d173），保留三维及 BEV 脉冲主干：

点云 → 3D 体素 MeanVFE → VoxelBackBone8x_3dv_snn → 高度压缩 → BaseBEVBackbone_spike → OpenCOOD 分类/框回归/方向头。

**这是单阶段适配版，不是 PointPillars，也不是完整 Voxel R-CNN 复现。** 未移植 RoI 精修。
计数激活为 floor(clamp(x,0,4)+0.5)，当前没有跨帧膜状态；稀疏坐标和计数仍是 int32/float32 张量，没有通信编码。
参数 3,020,564；体素 [0.2,0.2,0.1]，输出网格 50×126，两个朝向 anchor。

公开 kitti.pth 未用作 SNN 预训练：先前本地审计仅 59/341 个状态项兼容当前官方 SNN，ANN VoxelRCNN 则可匹配 257/257（spconv 布局转换后）。
该结果说明它不适合直接初始化当前 SNN，不等于权重对任何结构都不可用。
正式 SNN 从随机初始化开始，没有沿用单帧过拟合权重。

## 3. SNN 已完成的验证与训练

- 真实车路数据准备探针通过，但该探针并不是检测器性能结果。
- 完整适配检测器单帧 200 步：loss **2.5967 → 0.01019**，主干和头部关键梯度有限且非零。
- 严格权重重载和 4 样本 batch 通过，原始框解码 [4,12600,7] 全部有限且尺寸为正。
- 修复新配置的浮点网格截断：1008 曾被解析为 1007；独立解析器统一 sparse grid 与 anchor grid。
- 首次正式训练遇空 PCD 明确失败；完成全量审计后重启，失败日志保留。
- 正式运行：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_vehicle_single_v2`，PID 947559，GPU 4，启动代码 f4dfb96。
- 固定 60 epochs，batch=4，每轮 1,170 batch；Adam lr=0.001，40/55 轮后各乘 0.1。
- 前 10 轮训练 loss 1.7276 → 0.8267；当时 AP50 最佳第 9 轮：AP30/50/70 = 78.11% / 74.59% / 48.51%。
- 本轮最终归档至第 16 轮，第 16 轮 AP30/50/70 = **80.39% / 77.21% / 55.45%**。这是阶段验证，训练仍进行中。

## 4. 已有 PointPillars 权重

存在可用资产 best@57：

- 本地：`C:\Workspace\OpenCOOD\checkpoints\codyntrust_repro\stage1_wide_single_best57\net_epoch_bestval_at57.pth`。
- 服务器：`/data0/chen/gzc/workspace/logs/logs/dairv2x/repro_dair_single_pointpillar_wide_for_codyntrust_cobevflow_2026_09_11_02_51_02/net_epoch_bestval_at57.pth`。
- 既有来源 SHA256：c5fdef2fcc2914a09bb527cee6723730e70850056e3115f3197122e69431efb9。
- 结构使用 `point_pillar_codyntrust_single` 对应 CoDynTrust ResNet 变体，不能只凭名称用另一种 ResNet 强行加载。
- 历史 reader 训练时随机选择车端或路端；它是单视角输入训练，但不是纯车端训练。
- 历史 AP50/70 约 69.35% / 61.11%，用的是 1,789 索引和两侧 GT 汇总，**不可直接放进当前对照表**。
- 本轮新评估使用 SingleDAIRVehicle 和同一有效清单，固定车端输入及车端 GT；保留 PointPillars 的 pillar [0.4,0.4,5] 和 stride=2。
- 新评估目录：`/data0/chen/gzc/workspace/diagnostics/pointpillar_best57_vehicle_eval_20260929`，GPU 6；仅评估，没有重新训练。
- **本轮完整评估已完成**，status=complete，严格加载成功，权重 SHA256 与既有记录一致。
- 1,738 / 1,738 帧，22,971 GT 框，26,714 预测框；**BEV AP30/50/70 = 85.03% / 82.80% / 73.65%**。
- 新结果高于历史口径不能解释成模型进步：权重相同，样本分母及 GT 定义发生变化。
- 参数量 **30,516,500**，约为当前 SNN/ANN 的 10.1 倍；且历史训练用过两种单视角，故它是现有权重参考，不是受控同结构基线。

## 5. 同结构 ANN：现在启动的理由与控制

现有 GPU 有余量，SNN 数据、模型和预算已固定，应该立即并行建立直接对照，不必等 SNN 训练完。
ANN 重新从头训练；不复用 SNN 或 PointPillars 权重。

- 对照实现：`e3dsnn_single_ann`；用普通 ReLU 替换 39 个 Multispike。
- 结构、通道、卷积、BN、检测头、loss、体素、数据、增强、seed=20260929、batch 和 60 轮预算均保持一致。
- 实测审计：配置除名称/模型入口/激活标记外一致；246 个状态项初始值逐项相同，参数量同为 3,020,564。
- 这个对照测量的是“有界离散 Count4 与普通 ReLU”的整体变化；不能把差异全部归因于舍入。若需分离截断与离散效应，可后续加 clipped-ReLU，对应新实验，不在本次自动扩展。
- 先完成同样的 200 步过拟合检查，再从新的随机初始化启动正式训练。
- 过拟合实测通过：loss **2.73980 → 0.001585**，主干/检测头关键梯度有限且非零；这个结果仅用于检查接线，不能推断泛化优劣。
- 正式 ANN 已启动：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_vehicle_ann_relu_v1`，GPU **5**，PID **1058885**，启动代码 **89a3446**；同样从随机初始化训练 60 epochs。
- 日志为该路径加 `.log`，PID 文件为该路径加 `.pid`；checkpoint 和 metrics 的语义与 SNN 一致。
- PointPillars 属于已有工程参考；同结构 ANN 才是当前 SNN 的受控直接对照。完整 Voxel R-CNN 留作后续两阶段精度参考。

## 6. 证据和跟进规则

仓库证据目录：`C:\Workspace\OpenCOOD\OpenCOOD\docs\cp_snn\results`。
关键 JSON 同步至用户指定目录下的 `evidence/`：

- `pointpillar_eval_20260929.json`：新车端协议完整 AP、样本与框数。
- `pointpillar_manifest_20260929.json`：代码、配置、checkpoint SHA256 和参数量。
- `pointpillar_eval_status_20260929.json`：完整评估完成状态。
- `ann_equivalence_20260929.json`：ANN/SNN 初始参数和配置一致性检查。
- `ann_overfit_20260929.json`：200 步 loss 序列和梯度检查。
- `ann_training_manifest_20260929.json`：ANN 正式运行来源、随机种子和冻结配置。
- `snn_metrics_controls_start_20260929.jsonl`：启动对照时的 SNN 逐轮指标快照。
- `data_manifest_20260929.json`：完整有效 ID 与排除原因；另保留最初 SNN 过拟合和框解码证据。

运行说明：`C:\Workspace\OpenCOOD\OpenCOOD\docs\cp_snn\TRAINING.md`。
原始数据审计位于服务器 `diagnostics/e3dsnn_data_v1/`。

已有 Codex heartbeat `e3dsnn` 每 60 分钟检查本研究。新增 ANN 训练纳入同一个跟进，不创建重复任务。
SNN 运行目录是 `e3dsnn_vehicle_single_v2`，ANN 是 `e3dsnn_vehicle_ann_relu_v1`；两者均须检查，不因先完成其中一个就关闭跟进。
继续记录关键阶段指标、故障与修复、实际样本数、权重/配置/提交哈希。
SNN 和 ANN 均完成后分别重放最佳权重，再判断两者差异；单 seed 结果不当作统计显著性证据。
跟进中同步本文件至用户指定 cp-snn-plan，原有研究计划文档保持可追溯。
