# E-3DSNN 单端 DAIR 基线

## 模型和数据协议

汇报约定（2026-09-29 用户确认）：后续 **AP 默认指仓库 BEV AP**，AP30/50/70 对应 BEV IoU 0.3/0.5/0.7。
以此作为主指标，best 继续按 AP50 选择。体积 IoU 结果仅作为补充并明确写作 **3D AP**；不改变既有计算代码和历史结果标签。

本地 main 修改 → push → mindspore-184 的 OpenCOOD/main pull → GPU 运行。
本地路径：`C:\Workspace\OpenCOOD\OpenCOOD`。
远端路径：`/data0/chen/gzc/workspace/OpenCOOD`。

首版 **E3DSNN vehicle single-stage v1**：MeanVFE → 官方
VoxelBackBone8x_3dv_snn → HeightCompression → 官方 BaseBEVBackbone_spike →
OpenCOOD 分类/框回归/方向头。源代码固定于 E-3DSNN dbe5d173；保留 3D、2D 拓扑、
0..4 计数激活与替代梯度，归档 Apache-2.0 许可证及来源说明。
移植删除 debug 输出，使用显式 sparse residual feature 相加。

这是单阶段适配版，**不包含 VoxelRCNN RoI 精修，不等于完整原论文检测器复现**。
无跨帧膜状态、无协作模块、无预训练权重。连续检测头保留。
3D 最后共享候选层可以通过 return_spikes 返回 [M,128] 计数及 [M,4] bzyx 坐标；
模型内计数仍为 float32；独立 Count4 打包已完成 64 帧审计，见 PACKET_AUDIT.md。
尚未接入协同融合或测量实际能耗。

配置：`opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_single.yaml`。

- 车端当前点云及**车端本地标签**；不随机抽路端，不要求历史路端帧齐全。
- 原始 train.json / val.json（4,811 / 1,789）经全量车端 PCD 审计；使用 manifest 固定有效样本，不静默跳过、不丢尾 batch。
- 全量审计后训练 **4,678** 帧、验证 **1,738** 帧；分别排除 133 / 51 个读取为空的 PCD（包括空文件和损坏文件）。[固定划分及排除清单](results/data_manifest_20260929.json)。
- 范围 [-100.8,-40,-3.5,100.8,40,1.5]；体素 [0.2,0.2,0.1]，最多 70,000 体素。
- 3D grid xyz=[1008,400,50]，spconv z+1；3D 输出 stride=8，高度为 3，BEV 输入 384 通道。
- 检测网格 H×W=50×126，两个朝向 anchor；为浮点舍入新增独立配置解析，避免 1008 被截断成 1007。
- Adam，lr=0.001，weight_decay=1e-4；batch=4，60 epochs，40/55 轮 lr×0.1；梯度裁剪 10。
- 训练增强：x 翻转、±45° 旋转、0.95–1.05 缩放。验证无增强。
- 每轮完整验证，统计车端本地 GT 下的 **BEV AP@0.3/0.5/0.7**，不是 KITTI 3D AP。
- 不与旧 reader 筛选到 1,738 帧、或者使用两侧标签并集的历史 AP 直接比较。后续 ANN 对照必须使用同一新协议。

## 已验证

单帧 000010、24 个 positive anchors，200 步从 loss 2.5967 降到 0.01019。
首层稀疏卷积、末层稀疏卷积、BEV 卷积、回归头梯度均有限且非零。
原始计数范围为整数 0..4。[过拟合报告](results/overfit_20260929.json)。
两帧验证集 decode smoke 没有置信度超过阈值的预测，AP=0；不能当作泛化评估。
独立 audit 进一步检查 strict 权重加载、4 样本 batch 和不依赖分数阈值的 7D 框解码。
过拟合权重只用于接线验证，正式训练从新的随机初始化开始。

## 正式训练

```bash
cd /data0/chen/gzc/workspace/OpenCOOD
export PYTHONPATH="$PWD"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=4 /data0/chen/miniconda3/envs/opencood/bin/python -u \
  -m opencood.tools.train_e3dsnn_single --mode train --workers 4 \
  --output /data0/chen/gzc/workspace/diagnostics/e3dsnn_vehicle_single_v2
```

正式结果目录固定为上述 output，nohup 日志为同路径加 `.log`，PID 为同路径加 `.pid`。
`progress.json` 记录训练 batch 或验证阶段；`metrics.jsonl` 每个完整 epoch 一行。
只保留 `last.pt`（模型、优化器、scheduler、随机状态）和 `best.pth`（按完整验证 AP50 选取），
避免服务器有限磁盘被每轮 checkpoint 占满。last.pt 在完整验证后写入。
续训用相同命令加 `--resume`；必须先确认旧进程已结束，不能重复启动。
独立评估使用 --mode eval --checkpoint best.pth 和一个新的 output 目录。

4 样本 batch 与严格权重重载已通过，输出 [4,2,50,126]，原始解码框 [4,12600,7]，
全部有限且尺寸为正：[原始报告](results/batch_decode_20260929.json)。
首次正式运行（v1，代码 `12a5db0`，PID 941718）因 009369.pcd 是 0 字节文件而明确退出，失败日志保留。v2 在全量 PCD 审计后从随机初始化重启，使用 GPU 4；GPU 1 启动前检查发现已有其他任务，未在该卡启动。
v2 启动代码为 `f4dfb96`，PID **947559**，60 epochs、batch=4，每轮 1,170 个训练 batch。
代码快照为结果目录同路径加 `_source.tar.gz`。60 轮训练从随机初始化开始。

训练中不因读到新 main 提交自动重启，不改动运行中的模型/配置。每次执行保存配置、Git 提交和划分哈希。
数据异常或 NaN 会留下 status.json 并退出，修复后本地提交 push，再服务器 pull 并恢复。

## 定时跟进上下文

### 2026-09-29 首次定时检查

已完成 10/60 轮，每轮均覆盖 4,678 个训练样本和完整 1,738 帧验证集；进程正常，未重启。
训练平均 loss 从第 1 轮 1.7276 降至第 10 轮 0.8267。
按验证 AP50 选择的当前最佳为第 9 轮：BEV AP30/50/70 = 78.11% / 74.59% / 48.51%。
第 10 轮 AP50/70 = 73.44% / 49.06%；逐轮波动存在，继续原 60 轮计划，不据阶段指标调整协议。
`best.pth`、`last.pt` 均已实际生成。磁盘剩余约 79 GiB，暂无磁盘阻塞。
[前 10 轮原始指标快照](results/metrics_progress_20260929_2027.jsonl)。
这些是单阶段适配版、车端本地标签下的验证 BEV AP；尚未完成最终权重独立评估重放。

已创建本对话 heartbeat，ID `e3dsnn`，每 60 分钟，提示词“推进脉冲基线，无变化静默”（12 字）。

若训练仍在运行：检查对应 PID、日志、progress.json、metrics.jsonl 和磁盘即可，避免重复启动。
无状态变化时保持安静；完成、异常或有实质进展时更新本任务。
训练完成后：确认 status.json 为 complete 且进程退出，再读取 best_metrics.json，冻结 best.pth 和对应轮次元数据后严格加载，使用 `--mode eval --eval-3d --save-predictions --profile` 做完整验证重放，再用 CPU 离线复算（见下节）。同时归档 BEV AP、3D AP 与局限。当前训练仍按 BEV AP50 选 best，不将其称为按 3D AP 选出的最佳权重。
若进程失败：先定位并修复故障，按 main/push/pull 流程继续；保留失败日志。
用户随后授权测试现有 PointPillars 并推进同结构 ANN；同一 heartbeat 需跟进 SNN 与 ANN 两条已启动运行。
ANN：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_vehicle_ann_relu_v1`，PID 1058885，GPU 5，
配置 `opencood/hypes_yaml/dair-v2x/snn/e3dsnn_vehicle_ann.yaml`，60 轮。
PointPillars 评估：`/data0/chen/gzc/workspace/diagnostics/pointpillar_best57_vehicle_eval_20260929`，GPU 6。
新增关键进展同步至 `C:\Workspace\OpenCOOD\agent-doc\snn-plans\cp-snn-plan\04_单端基线与实验进展_20260929.md`，
其仓库镜像为 [RESEARCH_LOG.md](RESEARCH_LOG.md)。两条训练、最终最佳权重验证、逐帧归档和 CPU 精确复算均完成后停止本轮跟进。
该跟进不授权启动除此之外的新消融或协同模型长训练。

2026-09-29 用户要求继续推进研究后，已新增并完成第 19 轮 SNN 的 64 帧共享层 codec 审计，
详见 [PACKET_AUDIT.md](PACKET_AUDIT.md)。**用户随后要求先完善单车检测、后续再转入协同改造；以下最终共享层审计延后至协同阶段，不再作为当前单车阶段的完成条件。**届时在最终 SNN 最佳权重冻结后，使用同一清单再运行
`python -m opencood.tools.audit_e3dsnn_packet --checkpoint <固定权重> --output <新目录>`，
检验训练后期的通道稀疏度、实际特征包字节和无损检测输出是否仍成立。
该审计默认 64 帧、内部 1,200 秒上限；启动前确认空闲 GPU，保存进程日志并跟踪退出状态。
不把 feature packet 的字节数当作含位姿/时间戳、网络分片/重传的完整通信成本。

### 2026-09-29 补算 3D AP

默认推理链路 `inference.py -> eval_utils.caluclate_tp_fp -> common_utils.convert_format/compute_iou`
只计算 xy 多边形交并比，实质是 BEV AP。函数接受 (N,8,3) 三维框不等于使用体积 IoU。
已有 `pcdet_utils/iou3d_nms` 包不改变这一事实；默认 AP 链路没有调用其中的 3D IoU。

新增 `--eval-3d` 仅用于独立 eval，在同一批输出框上同时计算两种指标。
三维框须为无 roll/pitch 的直立框；体积交集 = 旋转矩形交集面积 × z 区间交集高度。
保留现有全局置信度排序、逐帧贪心匹配、VOC2010 精度包络积分；这不是 KITTI R11/R40。
默认 BEV 评估和运行中的训练均保持原协议。

启动评估前运行 `python -m opencood.tools.audit_e3dsnn_3d_ap --output <audit.json>`。
合成几何、重复匹配、空输入检查及现有 CUDA 算子交叉校验均须通过。
CUDA 的 check_in_box2d 使用 1 厘米边界容差，补算使用精确多边形相交；审计明确记录这些边界例外。

本次回放目录：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_3d_replay_20260929`。
`replay_manifest.json` 保存每条运行的命令、GPU、PID、源文件及快照 SHA256、轮次与历史 BEV 指标。
快照：PointPillars best@57、SNN last@19、ANN last@2；SNN/ANN 是阶段结果，不能用于最终优劣判断。
PointPillars 和 SNN 分别使用 GPU6/7；ANN 等 PointPillars 完成后复用 GPU6。
各模型输出子目录中 `evaluation.json` 的 `ap` 是 BEV，`ap_3d` 是三维体积 IoU AP。

## 单车推理归档与离线复算

评估必须使用固定权重文件和全新的输出目录。正在训练时，使用一次读取的原子 `last.pt`，
从同一对象提取 epoch、模型和配置哈希；不要将训练中分别更新的 best.pth/best_metrics.json 拼成未经验证的快照。
最终比较等待两条原定 60 轮训练结束，各自按验证 BEV AP50 选 best；不据中途结果调参或更换协议。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU> /data0/chen/miniconda3/envs/opencood/bin/python -u \
  -m opencood.tools.train_e3dsnn_single --mode eval --workers 2 \
  --config <对应模型yaml> --checkpoint <冻结的pth> --output <全新目录> \
  --eval-3d --save-predictions --profile
CUDA_VISIBLE_DEVICES='' /data0/chen/miniconda3/envs/opencood/bin/python -u \
  -m opencood.tools.replay_e3dsnn_predictions --evaluation-dir <同一目录>
```

保留每帧 NPZ（frame_id、预测角点、置信度、GT）、index.json 的顺序和逐文件 SHA256；
evaluation.json 固定索引哈希。离线工具在 CPU 上校验全部文件并按原 dtype、原帧序重算 AP，
要求 BEV / 3D AP、预测框总数、GT 总数与在线评估完全相同，生成 offline_replay.json。
该精确性指同一份已保存预测的复算；独立 GPU 推理仍可能受稀疏算子及体素内点顺序影响。
原始大文件保留在服务器，Git 和用户 evidence 目录归档指标、来源、哈希及复算报告。

计时采用 batch=1、CUDA 同步；剔除前 20 帧后汇报 H2D、模型前向、后处理及三者合计的均值/中位数/p95。
后处理包含仓库 decode/NMS/GT 整理。合计不含数据读取与预处理、AP 计算、NPZ 写盘；
另列的 batch_wait 来自预取 loader，不能当作完整数据预处理成本。
服务器上其他训练并行，当前浮点 CUDA 实现的这些计时不代表独占设备延迟或脉冲硬件能耗。

CPU 正确性回归：`python -m unittest opencood.tools.test_e3dsnn_prediction_replay -v`，
覆盖 BEV/3D 区别、重复框、漏检/空帧、NPZ/索引篡改和错误汇总 AP。

## 2026-09-30：SNN 已完成，继续跟进 ANN

SNN 原运行 `e3dsnn_vehicle_single_v2` 已正常完成全部 60 轮，原 PID 947559 已退出。
以 status.json=complete 与完整 metrics 为准；旧 progress.json 停留在 validating epoch60，不代表仍在验证。
不得重启这条已完成训练。按 AP50 选出的 best 为 epoch53，已冻结到：
`/data0/chen/gzc/workspace/diagnostics/e3dsnn_final_snn_20260930/snn_best_epoch_53.pth`。
该目录的 `pipeline_status.json` 跟踪最终 GPU 推理和 CPU 复算，`evaluation/` 保存逐帧 NPZ 与指标；
检查此处的完成状态后归档结果，避免重复启动。结果汇总见 [FINAL_BASELINE.md](FINAL_BASELINE.md)。
本轮最终 SNN 流程已完成，CPU 全量复算通过；最终 AP30/50/70 为 84.32% / 81.93% / 68.19%。
证据已归档到 `results/final_snn_20260930/`，后续 heartbeat 不需重复 SNN 验证，仅继续剩余 ANN 工作。

ANN 原运行 `e3dsnn_vehicle_ann_relu_v1` 继续训练。只在其 status=complete 且原进程退出后，
冻结对应 AP50 best，按相同流程执行最终验证和 CPU 复算。两条最终结果完整后汇总单车对照并结束当前跟进，
不自动启动协同模型训练。

## 2026-09-30：单车阶段全部完成，定时跟进已暂停

以上“继续跟进 ANN”现已完成。ANN 正常完成 60 轮，原 PID 1058885 已退出；按 AP50 选择 epoch55。
固定权重和最终结果目录为 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_final_ann_20260930`。
其 pipeline_status=complete，1,738 帧 GPU 推理、逐帧归档和 CPU 精确复算均通过。
ANN 最终 AP30/50/70 = 85.20% / 82.96% / 72.20%；SNN 为 84.32% / 81.93% / 68.19%。
汇总见 FINAL_BASELINE.md 和 `results/final_comparison_20260930.json`。
两条最终验证都无需重复运行；heartbeat `e3dsnn` 已通过应用工具设为 PAUSED。
当前单车阶段完成，不自动开始协同改造或额外长训练。

## 2026-09-30：用户批准进入协同融合阶段

上述“暂不开始协同”限制已被用户本次实施指令更新。单车结果保持冻结，不重训单车。
SNN/ANN 的三维消息、接收端 SE(3) 对齐、HeightCompression 后 BEV 融合现已实现，
13 项本地 CPU 检查及合成 CUDA 前后向、无损消息检查通过。服务器 SSH 三次连接超时，
尚未执行新配对数据审计、真实过拟合或融合训练；当前没有协同 AP 结果。
配对 GT、固定单车初始化、两组 30 轮预算及完整执行命令见 [FUSION.md](FUSION.md)。
既有 60 分钟 heartbeat 将用于推进融合阶段，状态无变化时静默。

VPN 恢复后，真实配对数据审计和两组 200 步过拟合均通过；正式 30 轮融合训练已启动。
SNN / ANN 分别使用 GPU 1/2，GPU 3/4 执行各自 ego/Max 固定权重对照与复算。
运行根目录 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_fusion_20260930_v1`，完整证据和进程入口见 FUSION.md 文末。
后续继续这些运行，不重复启动；AP 结果只有全量推理与复算完成后才归档为对照结果。

## 2026-09-30：首轮融合训练与最终验证全部完成

两组均完整完成 30 轮，原进程已退出，按 AP50 选择的 best 均为 epoch27。
固定权重在 `e3dsnn_fusion_20260930_v1/{count4,relu}_final/`，对应完整消息推理与 CPU 复算均为 complete。
SNN 最终 AP50/70 = 72.55%/54.26%，ANN = 73.80%/58.46%；平均实际消息 76.79/321.45 kB。
完整结果、GT 匹配和比较边界见 [FINAL_FUSION.md](FINAL_FUSION.md)。无需重训或重复最终推理。
heartbeat e3dsnn 已暂停；同预算 ego 微调和等码率量化对照为后续实验，当前尚未启动。
