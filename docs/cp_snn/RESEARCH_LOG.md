# 协同感知与 SNN：单端基线关键记录

日期：2026-09-29。此文件同步至用户指定的
`C:\Workspace\OpenCOOD\agent-doc\snn-plans\cp-snn-plan\04_单端基线与实验进展_20260929.md`。
代码在 `C:\Workspace\OpenCOOD\OpenCOOD` 的 main 修改、push，服务器 main pull。
本记录区分实测结果、进行中的训练和计划，不把阶段验证当作最终结论。

## 1. 数据与评估协议

- **汇报口径（用户于 2026-09-29 确认）：本研究后续未加限定的 AP 均指仓库默认 BEV AP。** AP30/AP50/AP70 分别对应 BEV IoU 阈值 0.3/0.5/0.7；进展汇报、主要对照表和最佳权重选择均沿用此口径。论文或报告的评估协议首次出现时明确说明采用 BEV IoU 与仓库 AP 积分规则。
- 已补算的三维体积 IoU 指标作为补充，始终显式标注 **3D AP**；历史证据保留原标注。引用其他论文的 AP 时保留其原始定义，不将其自动解释成本研究的 BEV AP。
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

## 7. 默认指标核实与补算 3D AP（2026-09-29）

用户要求确认“仓库是否默认计算 3D AP”。实际源码结论：**当前默认计算 BEV AP**。
`opencood/tools/inference.py:277` 调用 `eval_utils.caluclate_tp_fp`；
该函数经 `common_utils.convert_format` 只取前四个角点的 x、y，随后计算 Shapely 多边形交并比。
尽管模型输出 (N,8,3) 三维框，默认 AP 匹配没有使用 z 或高度。
`pcdet_utils/iou3d_nms/iou3d_nms_utils.py` 中存在体积 IoU 算子，但默认 AP 调用链没有调用它。
这个口径来自仓库既有通用评估代码，并非本次 SNN 移植将 3D AP 改成了 BEV AP。

新增 `eval_3d_utils.py` 和评估入口 `--eval-3d`，在同一批预测框上同时报告 BEV AP 和 3D AP。
三维 IoU 按直立旋转框的“底面积交集 × 高度交集 / 体积并集”计算。
匹配仍为逐帧置信度降序、一对一贪心匹配；AP 仍使用全局置信度排序和 VOC2010 精度包络积分。
这不是 E-3DSNN 论文表 3 的 KITTI Car 3D AP R11；跨数据集和跨 AP 定义仍不能直接比较。
原有训练进程继续运行，当前最佳权重仍按 BEV AP50 选择；训练完成后用同一最佳权重补报两种 AP。

校验：合成高度分离/部分重叠、水平分离、旋转框、重复预测、空输入及 AP 积分检查通过。
独立对照已有 CUDA 3D IoU：2,304 对随机框在 0.3/0.5/0.7 阈值上判定全部一致。
CUDA 内核对角点入框使用 1 cm 容差，有 27 对框落在该边界区间；全部框最大绝对差 0.00071235，
其余 2,277 对最大绝对差约 1.30e-7。补算采用精确多边形相交，不继承该 CUDA 容差。
未将首次过严的 CUDA 数值一致性断言失败隐去；查明内核容差后，按独立几何条件标记边界样本，
其余样本维持严格校验，并额外检查所有样本的阈值判定一致性。

回放目录为 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_3d_replay_20260929`。
SNN/ANN 从原子保存的 last.pt 提取最近完整 epoch 的模型，冻结为独立快照并记录 SHA256；
PointPillars 冻结既有 best@57。模型轮次分别为 PointPillars 57、SNN 19、ANN 2。
后两者是不同训练阶段的检查点，不是同预算最终对照，也不是按 3D AP 挑选的最佳检查点。

**三组完整补算均已完成**：各 1,738 / 1,738 帧、22,971 个 GT，无跳帧，严格加载权重。
以下百分数均来自本次回放的同一批预测框，AP30/50/70 指 IoU 阈值 0.3/0.5/0.7。

| 模型与固定轮次 | BEV AP30 | BEV AP50 | BEV AP70 | 3D AP30 | 3D AP50 | 3D AP70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointPillars best@57 | 85.03% | 82.80% | 73.65% | **84.34%** | **78.13%** | **49.10%** |
| E-3DSNN 适配版 last@19 | 81.04% | 78.03% | 57.97% | **80.24%** | **70.27%** | **24.51%** |
| 同结构 ReLU ANN last@2 | 63.17% | 55.12% | 17.21% | **61.82%** | **43.33%** | **4.18%** |

PointPillars 预测框数 26,714，三项 BEV AP 与前次独立评估逐项完全相同。
其 BEV AP70 与 3D AP70 相差 **24.55 个百分点**；SNN 对应差值 **33.46 个百分点**。
因此较高 BEV AP 不能作为三维检测精度已经足够的依据；增加垂直位置和高度约束后，高 IoU 阈值下的匹配明显减少。
SNN/ANN 尚在训练，且这里轮次不同，不能根据本表认定最终 SNN 优于 ANN。

SNN 本次预测框 26,646，训练内第 19 轮记录为 26,643，BEV AP 各项差异小于 0.005 个百分点。
训练内验证与独立回放未共享 RNG 状态，reader 在测试时仍打乱点序；本次使用固定 seed 的独立回放结果，
不把这类微小差异解释成模型改进，也不拼接旧 BEV AP 与新 3D AP 作为同次评估。

证据位于 `results/`，并同步至用户指定 `cp-snn-plan/evidence/`：

- `3d_ap_audit_20260929.json`：几何、匹配、AP 及 CUDA 对照检查。
- `3d_ap_replay_manifest_20260929.json`：三组固定快照、源文件哈希、epoch 和启动命令。
- `pointpillar_3d_ap_20260929.json`、`snn_3d_ap_20260929.json`、`ann_3d_ap_20260929.json`：完整指标。
- 各模型同前缀的 `*_manifest_20260929.json` 和 `*_status_20260929.json`：实际执行来源与完成状态。
- `3d_ap_summary_20260929.json`：核对数据清单、样本/GT 数、权重哈希后的合并结果。

## 8. 继续推进：共享层无损打包已实测（2026-09-29）

两条正式训练正常继续。本轮归档 SNN 至完整第 22 轮、ANN 至完整第 6 轮；
最新归档 AP30/50/70 分别为 SNN 80.91%/77.45%/55.28%，ANN 77.25%/73.50%/51.42%。
这是不同轮次的阶段状态，继续原 60 轮预算，不据此判断最终优劣。
逐轮证据为 `results/snn_metrics_packet_audit_20260929.jsonl` 和 `results/ann_metrics_packet_audit_20260929.jsonl`。

利用等待训练的时间，完成单端原生共享层的实际 Count4 codec 和 64 帧无损回放。
固定使用已有 SNN 第 19 轮快照，不使用训练中变化的 best 文件。
优化版实测 93.24% 通道值为零，11.83% 活跃位置为全零向量；
平均包 155.30 KB，相同元数据/地址的原始 FP32 包为其 22.84 倍，
同样剔除全零位置后的 FP32 和 uint8 包分别为其 20.13 和 5.21 倍。
这些是同一 SNN 特征的存储对照，**不是相对于量化 ANN 的协作收益**。

64 帧全部通过特征、三个检测头及最终框/分数精确一致性检查。
固定单帧的 40 次交替成对基准中，优化将 CPU 编码中位数从 26.75 ms 降至 10.65 ms，
包字节完全相同。未计 GPU 拷贝、完整网络传输或融合，不声称端到端时延/能耗优势。
两次独立 GPU 回放有微小计数差异，包总量相差 66 B；详见报告，不声称整个 GPU 链路完全确定性。

详细报告：[PACKET_AUDIT.md](PACKET_AUDIT.md)，同步到用户目录第 05 份记录。
下一步先完成单端公平对照，再建立 3D 地址的物理坐标契约及同字节量化 ANN 对照；
已发现末端 stride_zyx=[16,8,8]，跨车对齐不能误将 BEV stride=8 用于 z 轴。
当前仍无协同融合模型，尚未开始该模型的长训练。

## 9. 架构核对和候选共享层大小（2026-09-29）

按用户要求区分论文概念与实际检测代码。论文写下采样 kernel=2，但官方检测分支及当前代码实际为
kernel=3、stride=2、padding=1；基础块出口是连续残差，不能把所有层都称作 0..4 脉冲。
GPU 浮点实现不能直接宣称推理全部为稀疏 AC。

同一第 19 轮权重和 64 帧新增候选层测量：早期脉冲包平均 202.50 KB，
3D conv_out 末端脉冲包 155.30 KB，HeightCompression 后计数包 188.47 KB。
HeightCompression 是高度折入通道的 reshape，不减少数值数量；包大小差异来自稀疏组织方式。
建议优先共享 conv_out 最终 Multispike 后的 3D 计数与地址，再以 BEV 共享作为对照。
完整尺寸、实现差异和局限见 [SHARING_LAYERS.md](SHARING_LAYERS.md)，同步用户目录第 06 份记录。

## 10. 155 KB 的剩余压缩空间（2026-09-29）

93.24% 零元素比例和 155.30 KB 来自同一 conv_out 特征，后者已使用其稀疏性，不能重复扣除零比例。
逐帧核算平均地址/掩码/计数/包头分别为 36.22/96.58/22.23/0.276 KB。
非零值用 2 bit、压缩坐标并采用空间占用位图，按当前 64 帧数据估计可到 114.04 KB，尚未实现该新格式。
对已保存的单帧 128.19 KB 包，zlib level=1 实测压至 63.49 KB，30 次均能逐字节还原，
额外 CPU 压缩/解压中位数为 3.63/0.95 ms。这个单帧结果不是新的 64 帧平均值。
详细核算、其他压缩等级和局限见 [PACKET_AUDIT.md](PACKET_AUDIT.md) 第 6 节；默认 codec 暂未改动。
