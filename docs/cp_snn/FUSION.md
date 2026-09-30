# 车路融合实现与验证（2026-09-30）

用户已批准进入协同阶段。单车 SNN / ANN 的 60 轮基线保持冻结；新实验从各自最佳权重初始化。
当前状态：VPN 恢复后已同步服务器，完成真实配对数据审计和两组过拟合验证；SNN/ANN 各 30 轮融合训练正在运行，ego/Max 全量对照与复算已完成。
下文“本地合成验证”记录保留原阶段边界；最新服务器结果见文末。

## 架构与比较协议

发送端 3D 主干末端 `[128,3,50,126]` → 坐标/特征/元数据消息 → 接收端完整三维变换与三线性采样
→ HeightCompression 重排为 `[384,50,126]` → BEV 融合 → 原 BEV 主干和检测头。
HeightCompression 只是把高度放入通道，没有求和或丢弃高度；发送位置和融合位置因此分别选择在它的前、后。

三维输出格子的物理中心起点为 `[-100.7,-39.9,-2.65]` 米，步长为 `[1.6,1.6,1.6]` 米。
按卷积核、步长和 padding 推导中心，不能直接把输出数组索引当作原始点坐标。
接收端对每个目标格子反查源坐标，保留 roll/pitch/z；覆盖掩码仅表示发送端网格定义域，不能解释成观测占用或可见性。
三线性插值产生连续值；SNN 融合输出重新经过 Count4。这部分仍是浮点 GPU 运算，不能声称全流程纯 AC。

SNN / ANN 使用同一代码和数据协议，分别采用 Count4 / ReLU，编码器在车路之间共享参数。
每种激活提供三份配置：`e3dsnn_pair_{count4,relu}_{ego,max,residual}.yaml`。

| 模式 | 用途 | 行为 |
| --- | --- | --- |
| ego | 同一配对样本和协同 GT 下的车端对照 | 不编码路端，直接复用单车链路 |
| max | 无学习融合参照 | 对齐路端和车端逐元素取最大，再激活 |
| residual | 待训练主方法 | 拼接两端 BEV 与 3 个高度覆盖通道，用 1×1 卷积、GroupNorm、激活、1×1 卷积预测残差 |

残差形式为 `activation(ego + scale * coverage * delta)`，scale 初始 0.1；无覆盖区域严格保留 ego。
最后一层用小幅非零初始化，使前层能够获得梯度。训练中以 0.1 概率丢弃路端消息。
显式 disable_road 与原单车输出匹配；这不等价于“任意零值消息都必然保持原输出”。
完整模型参数双方均为 3,094,997，比单车增加 74,433。ego/max 配置也实例化相同模块，但不执行残差网络。

## 数据与消息边界

新 reader `PairedDAIRFusion` 使用两端当前配对点云，统一使用 cooperative world GT 投影到车端，保留 Car/Van/Bus/Truck。
这些 GT 与此前车端 local GT 不同，必须在新协议下重测 ego，不能直接拿旧 AP 相减宣称协同收益。
全量审计固定有效帧清单、原始划分和配对元数据哈希，并记录真实两端时间戳；不人为增加延迟，不表示物理同步。
共同增强同时作用于两个局部坐标系和 GT，并共轭变换相对矩阵。验证不随机打乱点顺序。

路端安装高度与车端不同，而现有 3D 主干只有 5 米输入高度窗。
当前实现用标定得到的 `T[2,3]/T[2,2]` 平移路端局部 z 原点，同时精确修正发送端到接收端变换。
它不使用 GT、不预先旋转 XY，但依赖接收者，因此不是可直接广播给任意车辆的统一表示。
真实数据审计必须检查平移分布、平移前后裁剪点数及有效重叠；尚未验证该策略在 DAIR 全量数据上是否合适。

推理可启用真实 bytes 的进程内序列化/解码：SNN 复用无损计数打包，ANN 使用坐标与 FP32 特征的压缩 NPZ。
元数据包含帧、真实时间戳、三维形状、物理格子定义和相对矩阵；接收端实际消费解码结果。
训练直接使用可微特征，不经过 CPU 字节序列化。未实现网络套接字或异步传输。
ANN 当前是保留原精度的参照，尚不是等比特率量化对照；不能将两者包大小差全部归因于 SNN。

## 已完成验证

本地 opencood 环境：Python 3.7 / PyTorch 1.10+cu113 / spconv，RTX 4060 Laptop 8GB。
13 项 CPU 检查通过，覆盖物理中心、恒等/平移/roll/边界与插值梯度、两种消息编解码、
高度原点变换、共同增强、合成配对 reader/GT/collate，以及原预测归档与 AP 复算回归。

合成 CUDA 检查同时运行 SNN 和 ANN 的真实网络：batch=2、无路端回退、Max 使用路端、非零特征无损消息往返，
以及 3D 主干、BEV 主干、融合层和检测头的有限非零梯度均通过。
消息一致性检查固定同一份非零主干特征，隔离独立稀疏 CUDA 执行的数值波动。
[原始 GPU 报告](results/fusion_synthetic_20260930.json) 中的包大小仅来自合成随机输入，不是 DAIR 结果。
这些检查不代替真实数据训练验证；目前没有协同 AP、最终通信量或能耗结论。

```bash
python -m unittest opencood.tools.test_e3dsnn_fusion opencood.tools.test_paired_dair_fusion opencood.tools.test_e3dsnn_prediction_replay -v
python -m opencood.tools.audit_e3dsnn_fusion --output <新的报告路径>
```

## 连接恢复后的执行顺序

先检查服务器仓库状态，确认没有冲突，再在 main 执行 `git pull --ff-only origin main`。
复用 `/data0/chen/miniconda3/envs/opencood/bin/python`，设置 PYTHONPATH 为服务器仓库；
`NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
运行前检查 nvidia-smi，选择空闲卡；每次实验使用全新输出目录，记录日志、PID、提交及源文件快照。
失败不覆盖历史输出，不自动跳过数据或梯度错误。

1. 先执行全量两端数据审计，检查两个 split 的排除原因、裁剪点数、高度平移和时间差。
   如果大量无效或路端裁剪异常，先修复协议，不能直接带着筛选偏差启动长训练。

```bash
python -m opencood.tools.prepare_e3dsnn_fusion_data \
  --config opencood/hypes_yaml/dair-v2x/snn/e3dsnn_pair_count4_residual.yaml \
  --output /data0/chen/gzc/workspace/diagnostics/e3dsnn_pair_data_v1 --workers 4
```

2. 两种 residual 模型分别执行真实单 batch 200 步过拟合、非零融合梯度核验、两帧解码。
   如预训练权重使初始 loss 已很低而既有“下降一半”门槛失败，先检查具体 loss/预测，不能隐瞒或直接绕过。

```bash
python -m opencood.tools.train_e3dsnn_single --mode overfit --workers 0 --steps 200 \
  --config <对应 residual YAML> --single-checkpoint <对应固定单车权重> --output <新目录>
```

固定 SNN 权重：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_final_snn_20260930/snn_best_epoch_53.pth`，
SHA256 `404dfa3aec12b657a7450741974cec7b7af62b1b5e35696dcba37ff42709cdbd`。
固定 ANN 权重：`/data0/chen/gzc/workspace/diagnostics/e3dsnn_final_ann_20260930/ann_best_epoch_55.pth`，
SHA256 `971575638d7fa2bf8a6de4e3c7920e5499eb0ada9dfc2e9b6c6ca5cb450d3079`。

3. 两种模型分别在新配对验证集执行 ego/max 全量评估，Max 启用 `--packet-roundtrip`；保存预测并 CPU 复算。

```bash
python -m opencood.tools.train_e3dsnn_single --mode eval --workers 2 \
  --config <ego 或 max YAML> --single-checkpoint <对应固定单车权重> \
  --output <新目录> --eval-3d --save-predictions
# max 另加 --packet-roundtrip；输出真实 packet_bytes 的均值/p95/总量。
CUDA_VISIBLE_DEVICES='' python -m opencood.tools.replay_e3dsnn_predictions --evaluation-dir <同一目录>
```

4. 数据和真实过拟合检查通过后，按相同预算训练两种 residual 模型：batch=2、30 epochs、Adam lr=0.0002，
   前 3 epochs 冻结预训练部分参数与 BN，之后 27 epochs 联合微调；第 20/27 轮后乘 0.1。
   种子均为 20260929；各自继承已冻结的单车最佳权重。该预算是首轮实验设定，尚无充分性结论。

```bash
python -u -m opencood.tools.train_e3dsnn_single --mode train --workers 4 \
  --config <对应 residual YAML> --single-checkpoint <对应固定单车权重> \
  --warmup-epochs 3 --output <新训练目录>
```

恢复训练用同一配置、目录及 `--warmup-epochs 3 --resume`，省略 `--single-checkpoint`。
训练完成后按验证 BEV AP50 选择并冻结 best，使用 `--checkpoint`、`--packet-roundtrip`、
`--eval-3d --save-predictions` 完整评估并 CPU 复算。AP 默认指 BEV；3D AP 单独列出。
在同一配对协议内比较 ego → max → residual；另外区分不同激活、编码方式及训练预算，避免混淆收益来源。

后续由既有 60 分钟 heartbeat 跟进连接与运行状态；无变化静默，有真实结果、失败或需要用户处理时再通知。

## VPN 恢复后的实测与启动记录

2026-09-30 连接恢复后，全量检查 train 4,811 帧、val 1,789 帧，分别保留 **4,678 / 1,738** 帧。
排除 **133 / 51** 帧，原因均为两端原始点云至少一端为空或无效；没有裁剪后为空造成的额外排除。
此处有效数量与单车阶段相同，但 GT 已改为协同世界标签，不能沿用旧 AP 作直接对照。
路端 z 平移范围：train −3.123～2.129 米、val −2.232～1.834 米；
平移后路端保留点数的最小值分别为 4,919 / 7,275，中位数为 34,715.5 / 35,428。
每个 split 等距抽查 32 帧三维网格重叠，所有样本有正覆盖，覆盖比例中位数为 45.96% / 40.56%。
这是网格定义域重叠，不是传感器可见率。真实两端时间差绝对值中位数约 15.38 / 14.84 ms。

两种 residual 模型均完成真实单帧 200 步过拟合及两帧解码，33 个正锚框；
3D/BEV 主干、检测头及融合网络首末卷积的梯度均有限且非零。
SNN 前/后 10 步平均 loss 为 **0.690055 → 0.045444**，ANN 为 **0.647846 → 0.000650**，均通过既定下降门槛。
两帧解码仅检验链路，不将其中 AP 当作全量指标。

正式训练重新加载各自冻结的单车最佳权重，不使用过拟合产生的权重。
两组统一 batch=2、30 epochs、warmup=3、Adam lr=0.0002，GPU 1 为 SNN、GPU 2 为 ANN。
首轮训练已实际推进，记录时双方均完成至少 101/2,339 个 batch，loss 与融合训练梯度范数有限且非零。
GPU 3/4 分别执行 SNN/ANN 的 ego 全量评估 → CPU 复算 → Max 消息往返全量评估 → CPU 复算。
对照评估与训练并行，始终使用固定单车权重。当前尚无完整协同 AP 结果。

所有运行位于 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_fusion_20260930_v1`：

- `count4_train/`、`relu_train/`：progress、metrics、last/best 和最终 status。
- `count4_training_pipeline.json`、`relu_training_pipeline.json`：监督进程和实际子进程 PID、完成/失败状态。
- `count4_controls_pipeline.json`、`relu_controls_pipeline.json`：对照当前阶段与完成状态。
- `*_plan.json`、`launch.json`、`source.tar`：命令、GPU、代码提交、源码快照和超时约束。
- `count4_overfit/`、`relu_overfit/`：真实过拟合损失曲线、梯度与解码检查。

过拟合硬超时 30 分钟，单项对照/复算 1 小时，单组训练 24 小时；失败写入 pipeline 状态并停止该流程，不自动重试。
数据完整审计留在 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_pair_data_v1`。
[启动证据](results/fusion_startup_20260930.json) 汇总审计、过拟合、命令、PID 和首轮进度。
代码快照为 main `5807195`，包含融合实现 `6bb64c5`；其他研究提交不改变本轮 E-3DSNN 配置。
后续 heartbeat 读取这些既有运行，不重复启动；两组训练结束后冻结 best 并按前述流程完成最终消息推理和 CPU 复算。

## 2026-09-30 13:30：固定权重对照完成，融合训练进入第 11 轮

两组训练已完整完成第 1～10 轮，轮次连续；每轮覆盖 4,678 个训练样本、1,738 帧验证及 28,622 个协同 GT。
检查时 SNN 正在训练第 11 轮、ANN 正在验证第 11 轮，两个原训练进程存活，没有重启或调整预算。
ego/Max 四个固定权重对照全部完成，并通过 CPU 全量预测复算、全部文件哈希和 BEV/3D AP 精确一致检查。

| 激活 | 方法 | AP30 | AP50 | AP70 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| Count4 SNN | 仅车端，固定单车权重 | 69.82% | 66.62% | 54.77% | 全量推理与复算通过 |
| Count4 SNN | Max，固定单车权重 | 69.06% | 64.47% | 49.23% | 真实消息往返及复算通过 |
| Count4 SNN | 可学习 residual，第 10 轮 | 77.70% | 72.07% | 52.01% | 训练内阶段验证，尚非最终结果 |
| ReLU ANN | 仅车端，固定单车权重 | 70.45% | 67.46% | 57.98% | 全量推理与复算通过 |
| ReLU ANN | Max，固定单车权重 | 65.24% | 62.27% | 51.07% | 真实消息往返及复算通过 |
| ReLU ANN | 可学习 residual，第 10 轮 | 78.29% | 72.98% | 57.06% | 训练内阶段验证，尚非最终结果 |

AP 指 BEV，采用仓库 VOC2010 积分规则，不是 KITTI R11/R40。补充 3D AP 保存在原始报告。
相比同协议固定车端权重，直接 Max 的 AP50 分别下降 2.15 / 5.19 个百分点；
第 10 轮 residual AP50 分别高 5.45 / 5.52 个百分点，但 AP70 仍低 2.77 / 0.92 个百分点。
这说明当前宽松 IoU 阈值下阶段表现改善，尚不能声称严格框定位全面改善。
residual 相比固定 ego 同时增加了融合与协同 GT 微调；在增加同预算 ego 微调消融之前，不能把上述增幅全部归因于路端消息或融合模块。
保持当前预定 30 轮，不因这次中途指标改预算；同预算 ego 微调作为后续归因实验，当前未启动。

Max 对照实际解码全部 1,738 个路端特征包：SNN 平均 **73,366.78 bytes（73.37 kB）**，p95 91.91 kB；
ANN 平均 **268,868.54 bytes（268.87 kB）**，p95 319.38 kB，均包含特征和元数据，不含网络帧/重传。
这里 kB=1,000 bytes。SNN 使用计数编码，ANN 使用压缩 FP32 NPZ，尚非等码率对照；
这批结果使用冻结的单车权重编码路端，不能替代训练后的 residual 模型通信量，也不能直接与此前 epoch19 的约 155 KB 混比。

原始证据见 [13:30 运行检查](results/fusion_progress_20260930_1330.json)。
四个对照流程已完成，无需再运行；后续继续两组原训练，完成后冻结最佳权重并执行最终消息推理、归档及 CPU 复算。

## 2026-09-30 15:30：首次学习率衰减后的阶段结果

SNN 完整完成 23/30 轮、正在训练第 24 轮；ANN 完成 24/30 轮、正在验证第 25 轮。
两个原训练进程正常，指标轮次连续，所有已完成轮次的 train/val/GT 数均为 4,678/1,738/28,622。
按预定计划在第 20 轮后降低学习率；双方按验证 AP50 选择的 best 均更新至第 23 轮。

| 第 23 轮 AP50 best | AP30 | AP50 | AP70 | AP50 相较固定 ego | AP70 相较固定 ego |
| --- | ---: | ---: | ---: | ---: | ---: |
| SNN residual | 77.67% | 72.47% | 54.22% | +5.85 个百分点 | −0.56 个百分点 |
| ANN residual | 78.23% | 73.39% | 58.36% | +5.93 个百分点 | +0.38 个百分点 |

相较第 10 轮 best，SNN AP70 提高 2.21 个百分点、ANN 提高 1.30 个百分点；
这是衰减后观察到的变化，未做学习率因果消融。ANN 阶段 AP70 略高于固定 ego，SNN 尚略低。
这是训练内阶段验证，不是最终冻结权重的独立消息推理与 CPU 复算，也不改变融合/额外微调尚未分离的边界。
保持既定 30 轮预算与第 27 轮后再次衰减计划，不另行重训或重复完成的固定对照。
[本次完整逐轮快照](results/fusion_progress_20260930_1530.json) 保存全部已完成指标、快照哈希与运行状态。
