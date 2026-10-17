# 变化积累与缓存读出原型

2026-10-16：用户确认当前阶段以设计启发和工程可行性为主，不优先扩大多种子统计。本轮从已达到 GRU 精度水平的混合脉冲模型继续探索实际计算复用。

## 设计与边界

新增 `delta_gru`，保留 GRU 的三组门控/候选卷积及读出卷积、参数名称和初始化，不改 Pillar、主干和检测头。先将全部历史状态按同一 SE(2) 位姿对齐，再计算：

```
H = GRU_proposal(X, R)
V = U + (H - R)
S = Hstep(V/theta - 1) - Hstep(-V/theta - 1)
R_next = R + theta*S
U_next = V - theta*S
```

这里创新量为学习到的隐状态提议与已保存证据的差，而非原始点云几何残差。持续的小误差可累积触发正负事件；读出使用累积证据 R，不直接读取 U。固定阈值 theta=0.25，未按 AP 调参。

缓存包括 U（64）、R（64）、读出 Y（64）、刷新计时（1）、上一帧占据掩码（1）与有效标志（1），共 195 个 FP32 通道。在 200×504 网格上为 **78,624,000 字节**，是原 GRU/spike_gru 的 3 倍；FM 参数仍为 259,712，整模型 30,776,212。新增内存必须与后续省算收益一起评价，不再宣称同缓存对照。

无历史、新视野或距离上次强制刷新达到 1 秒时，R 直接取当前 H，U 清零。这是显式的模拟量初始化/刷新，不伪称脉冲更新。无点云返回本身不会清除 R。新事件、强制刷新或占据掩码变化的位置及其 3×3 输出邻域刷新 Y，其他位置沿用对齐后的 Y；计时变化自身不会触发 Y 刷新。因此该规则定义的是新的时序模型，不能声称与每帧完整重算的 GRU 数值等价。旋转/重采样可能改变卷积邻域，周期刷新用于限制陈旧缓存，仍须由检测实验检查拖影和误差。

第一版门控、候选、读出卷积和后续主干**全部稠密执行**，`where` 仅模拟读出复用。新增逐帧统计区分通道发放率、发生事件的空间位置比例、含卷积邻域的读出刷新比例和 8×8 活跃块比例。通道脉冲稀疏不代表空间块稀疏，也不等于实际延迟下降。

## 实验路径

- 配置源码：`opencood/hypes_yaml/dair-v2x/snn/int_ego_delta_gru.yaml`。
- 181 根目录：`/data1/tzm/workspace/diagnostics/int_delta_gru_20261016`；实际 YAML 与完整清单在 `inputs/`。
- 沿用原始 P1、完整 DAIR 清单、seed303、TBPTT4、首轮 FM 预热，保持无异步延迟。
- 先以 GRU/delta_gru 配对两轮工程试跑，核验学习、有限状态和精确恢复；通过后仅新训 delta_gru，复用已经完成的配对 GRU 作为方向性参照，避免无效重复训练。
- 正式训练至少 20、上限 40 轮，使用既有平台期停止规则；两轮小样本试跑仅作工程检查。
- 完成后先比较 AP 与读出刷新比例，若精度保留且块更新足够稀疏，再实现目标层的块执行。若更新几乎遍布全图，先修改机制，不投入稀疏算子工程。

32 项 INT 测试已在 181 通过，包括累计小误差双极触发、静止证据保留、周期刷新、3×3 邻域、位姿边界、场景重置、参数一致及跨帧梯度。

## 当前基线回放修复

GRU 历史回放完成。spike_gru 首次 aligned 回放与训练最佳轮 AP 相差约 0.000119683（0.01197 个百分点），触发严格检查；发现训练设置 `cudnn.deterministic=True` 而回放入口缺失。已统一设置并复跑，保留 `replay_spike_gru_failed_nondeterministic` 原证据，不放宽 AP 一致性阈值。以修复后回放是否通过判断原因，不能提前宣称异常已解决。

同卡顺序 profile 工具为 `opencood.tools.profile_int_memory`，在相同 GPU、固定 256 个验证扫描上测 Pillar/scatter/FM/backbone/head，并采样空间事件支持。它包含统计开销、排除数据 IO，仅作模块选择依据，不是端到端或能耗基准。

## 实测进展与定时接续

1. 统一确定性设置后，spike_gru 的 aligned 回放 AP70=0.759872411059917，与训练最佳轮精确一致，原偏差已排除；reset/no-align 仍在运行。旧失败目录保留。GRU aligned/reset/no-align 分别为 75.8736% / 74.3535% / 71.4649%。
2. 同一张 GPU 0 上顺序执行 256 个验证扫描（前 10 帧不计时）：GRU/spike_gru 模型墙钟中位数为 34.02/36.28 ms；FM 均值 5.15/7.39 ms，backbone 均值均约 21.2 ms，shrink 约 4.87 ms。包含插桩开销，不用于声称 SNN 固有延迟差。当前 spike_gru 通道发放率 22.09%，事件位置 75.64%，8×8 活跃块 91.42%，说明直接对当前脉冲做块跳算空间很小。证据：`int_spike_results/spiking_gru_20261016/profile_256.json`。
3. 新原型配对两轮工程试跑及精确断点恢复通过，全部参数/优化器/调度器/契约/AP 一致。第 2 轮小样本验证发放率 7.84%，事件位置 34.97%，读出刷新位置 58.91%，8×8 刷新块 76.65%，强制刷新位置 13.41%。这些是尚未训练充分的工程数据，不与上一项正式权重的不同样本作效果比较。
4. 已从原始 P1 启动 **delta_gru 正式训练**，181 GPU 2，初始 PID 17813，根目录如上，启动记录 `launch_full_delta_gru.json`，输出 `full_delta_gru`，日志 `full_delta_gru.log`。已确认首轮至少 384 帧、78 次更新，loss 有限。未重复训练完整 GRU。记录见 [delta_gru_20261016](int_spike_results/delta_gru_20261016/launch_full_delta_gru.json)。

每小时任务现在同时跟进：旧根目录的 `replay_spike_gru/status.json`，以及新根目录的 `full_delta_gru/{result,curve,failure}.json`、日志、GPU 和磁盘。旧回放完成后运行 `summarize_int_memory --root <旧根目录> --modes gru spike_gru --output <旧根目录>/paired_report` 并归档报告。delta_gru 中断时先检查原因、进程及空闲 GPU，再使用 `resume_int_memory --root <新根目录> --launch-record launch_full_delta_gru.json`；达到平台期后对最佳权重做完整 aligned/reset/no-align 回放及同卡 profile，再判断是否值得实现块级稀疏执行。暂不承诺实际加速，不进行多种子扩张。仅完成或异常通知。

最新接续：用户授权根据实测耗时将检查间隔缩至一小时以内。delta_gru 首轮训练 560.47 秒、验证 369.27 秒，合计约 15.5 分钟，当前进入第 2 轮；任务已改为每 **30 分钟**检查（后续可按剩余事件耗时调整），继续仅完成或异常通知。上段每小时设置由此替代。

旧 GRU/spike_gru 的六项回放现已全部完成且通过 `summarize_int_memory` 审计，报告见 [paired_report/report.json](int_spike_results/spiking_gru_20261016/paired_report/report.json)。不再重复启动旧回放，主要跟进 delta_gru 训练。

| 模型 | 对齐 AP70 | 清空历史 AP70 | 不对齐 AP70 |
|---|---:|---:|---:|
| GRU | 75.8736% | 74.3535% | 71.4649% |
| spike_gru | 75.9872% | 74.2929% | 71.5460% |

脉冲 GRU 的对齐历史收益约 +1.6944 个百分点，GRU 为 +1.5201 个百分点；本阶段支持继续研究脉冲记忆和创新更新，不需要先追加多种子精密消融。

## 2026-10-17 训练完成与诊断

delta_gru 在第 26 轮达到平台期，最佳第 20 轮 AP70 **73.2605%**，相比 GRU 低 2.6131 个百分点、相比 spike_gru 低 2.7267 个百分点。最后一次超过 0.001 AP 的显著改善在第 16 轮，因此第 26 轮 stale=10；第 20 轮刷新绝对最佳但不足以重置显著改善计数，这符合预设规则。最终学习率 1.25e-5。26 轮训练/验证覆盖、有限 loss/AP、停止条件及最佳权重哈希审计通过，见 [training_completion.json](int_spike_results/delta_gru_20261016/training_completion.json)。无需恢复训练。

最佳轮完整 5,794 帧验证中，通道发放率 16.39%，事件位置比例 46.55%，含邻域的读出刷新位置 67.05%，8×8 刷新块 82.23%，强制刷新位置 11.71%。缓存增至 3 倍且块刷新仍较密集，当前原型尚未呈现值得投入稀疏算子工程的精度—更新范围折中。

膜残差最大绝对值为 17.1564，明显高于 0.25 阈值。它仍有限且程序正常，但提示单步最多一次脉冲可能无法及时释放累积误差；这是后续诊断线索，尚未证明它是精度下降原因。后续优先检查持续偏差积累及读出陈旧误差，避免立即扩大训练或修改原实验。

已启动最佳权重 `replay_delta_gru` 的 aligned/reset/no-align 完整回放（GPU 2，初始 worker PID 6900），并在 GPU 0 对同一固定 256 帧运行 `profile_int_memory --modes delta_gru`。每 20 分钟跟进这两个后处理事件，替代训练期 30 分钟间隔；仅完成或异常通知。回放结束后运行 `summarize_int_memory --root /data1/tzm/workspace/diagnostics/int_delta_gru_20261016 --modes delta_gru --output /data1/tzm/workspace/diagnostics/int_delta_gru_20261016/final_report`，归档报告后再决定最小机制改动。

256 帧 profile 已完成：模型墙钟中位数 39.74 ms，FM 均值 9.02 ms，backbone 均值 21.81 ms，shrink 4.87 ms；该样本读出刷新位置 66.25%、8×8 刷新块 82.27%。证据：[profile_256.json](int_spike_results/delta_gru_20261016/profile_256.json)。此次依然是含插桩的稠密参考实现，不将不同时段的 profile 差值解释为受控加速对比。后续只需继续检查历史回放，无需重跑 profile。

## 2026-10-17 完整历史回放完成，进入固定权重机制诊断

三项回放全部通过逐帧顺序、覆盖、权重哈希及 aligned AP 一致性检查，报告为 [final_report/report.json](int_spike_results/delta_gru_20261016/final_report/report.json)。

| 模型 | 对齐 AP70 | 清空历史 AP70 | 不对齐 AP70 | 对齐历史收益 |
|---|---:|---:|---:|---:|
| GRU | 75.8736% | 74.3535% | 71.4649% | +1.5201 pp |
| spike_gru | 75.9872% | 74.2929% | 71.5460% | +1.6944 pp |
| delta_gru | 73.2605% | 72.7762% | 57.4270% | +0.4843 pp |

delta_gru 的历史收益明显较小且不对齐下降更大。这里 reset 同时改变膜、证据和读出缓存，不能将差异单独归因于一种状态。为选择下一版机制，已在固定第 20 轮最佳权重上启动两项完整验证干预，不重新训练：

- `instant_residual`：只将 V=U+(H-R) 改为 V=H-R，不再把先前误差重复积分；仍保存相同状态结构并执行脉冲更新与缓存读出。
- `dense_readout`：保持原误差积累和证据更新，每帧读取完整 post_fusion 输出，关闭 Y 的跨帧复用。

原默认行为保持不变，35 项 INT 语义/回归测试通过。两项不是重新训练的候选模型，AP 变化只能提供机制敏感性线索，不作为新模型能力结论。启动代码 `492c99b`，记录 [mechanism_probes_launch.json](int_spike_results/delta_gru_20261016/mechanism_probes_launch.json)。服务器根目录下 `mechanism_probes/{instant_residual,dense_readout}`，GPU 0/1，初始 PID 13309/13310，分别查看 `run.log`、`eval/result.json`、`eval/manifest.json`、`eval/eval_frames.jsonl`。完成后核验同一权重哈希与 5,794 扫描/1,738 标注/22,971 GT、顺序和配置干预，再比较 AP、残差最大值及刷新比例。

完整训练与历史消融已收尾，后续只跟进这两项机制诊断；每项预计约 6 分钟，检查间隔改为 **10 分钟**，替代前述 20 分钟。仅完成或异常通知；不重复训练已达平台期的旧模型，也暂不投入稀疏算子实现。

## 2026-10-17 机制诊断完成，选择即时状态差原型

两项固定权重诊断均已完成，对相同最佳权重、清单、逐帧顺序、5,794 扫描/1,738 标注/22,971 GT 和实际策略配置的检查通过。证据：[mechanism_probes_report.json](int_spike_results/delta_gru_20261016/mechanism_probes_report.json)。

| 固定权重策略 | AP70 | 相对原策略 | 读出刷新位置 | 8×8 刷新块 | 最大残差绝对值 |
|---|---:|---:|---:|---:|---:|
| 原误差积累＋缓存读出 | 73.2605% | — | 67.05% | 82.23% | 17.1564 |
| 即时状态差＋缓存读出 | 74.1284% | +0.8679 pp | 55.66% | 75.54% | 1.7500 |
| 原误差积累＋完整读出 | 73.2230% | -0.0375 pp | 100% | 100% | 17.1564 |

结果支持优先去除重复积分，不能据此证明全部误差来自它；关闭读出复用在这组权重上没有改善。下一步只训练一个候选：`residual_policy: instant`，保留 `readout_policy: cached`、阈值 0.25、1 秒强制刷新及所有其他协议。

即时状态差 V=H-R 仍通过 R 保留历史证据，但不再通过 U 积分误差；旧 U 通道暂时保留为诊断记录，不参与下一帧电压。为隔离机制，本轮仍保存 195 通道缓存，不能宣称内存优化或仍具有原版 U 的积分语义。若新训练有效，再考虑删去无用缓存及实际跳算。

新配置：`opencood/hypes_yaml/dair-v2x/snn/int_ego_delta_instant.yaml`。新根目录：`/data1/tzm/workspace/diagnostics/int_delta_instant_20261017`，模式名仍为 `delta_gru`，必须通过配置和根目录区分。先配对工程试跑并核验精确恢复，再从原始 P1 启动全量候选训练；不使用刚完成诊断的第 20 轮权重进行初始化。只沿用已有完整 GRU 对照，不再重复完整 GRU 训练。

**当前接续：** 即时状态差版本的两轮工程试跑与精确断点恢复均已通过，正式训练已在 181 GPU 0 启动（初始 PID 21490）。实际配置 `<新根目录>/inputs/config.yaml`，输出 `full_delta_gru`，日志 `full_delta_gru.log`，启动记录 `launch_full_delta_gru.json`。seed303、TBPTT4、首轮 FM 预热、原始 P1、至少 20/最多 40 轮及既有平台期规则保持不变。证据存于 [delta_instant_20261017](int_spike_results/delta_instant_20261017/launch_full_delta_gru.json)。

定时任务回到每 **30 分钟**检查新根目录，替代短诊断阶段 10 分钟；旧两项诊断已完成，不要重复启动。启动时磁盘约剩 14 GiB，继续监控每轮 5 GiB 安全下限。若新训练中断，查明原因后以 `resume_int_memory --root <新根目录> --launch-record launch_full_delta_gru.json` 接续；平台期后才选择最佳权重执行回放和 profile。当前 74.1284% 是旧权重干预结果，不是本次新训练的成绩。

## 2026-10-17 即时状态差训练完成（当前接续）

用户随后要求减少静默轮询，训练期间已改为每 2 小时检查。此次检查确认即时状态差版本正常完成 27 轮，最佳第 17 轮 AP70 **74.6976%**，stale=10，最终学习率 6.25e-6，满足预设平台期停止规则。全部 27 轮覆盖、有限 loss/AP、最佳权重哈希和实际 instant 策略审计通过，见 [training_completion.json](int_spike_results/delta_instant_20261017/training_completion.json)。不恢复训练。

| 版本 | 最佳验证 AP70 | 读出刷新位置 | 8×8 刷新块 | 最大残差绝对值 |
|---|---:|---:|---:|---:|
| 原误差积累 delta_gru | 73.2605% | 67.05% | 82.23% | 17.1564 |
| 即时状态差，重新训练 | 74.6976% | 55.06% | 75.11% | 1.7500 |

统计取各自最佳轮的完整 5,794 帧验证。即时版本较原误差积累版提升 1.4371 pp，但仍低于 GRU 1.1760 pp、spike_gru 1.2896 pp；缓存仍是 195 个 FP32 通道，块刷新仍较密集，不能据此宣布省算优势。此前 74.1284% 仅为旧权重干预成绩，须与此处重新训练结果区分。

已启动新根目录下 `replay_delta_gru` 的 aligned/reset/no-align 完整回放，GPU 1，初始 worker PID 36762；在 GPU 2 运行固定 256 帧 `profile_256.json`，不与早期 GPU 0 的不同时段 profile 作严格加速对比。历史回放预计约 18–25 分钟，仅在此后处理阶段临时调整为每 30 分钟检查，避免训练期频繁静默轮询。下一步核验 `replay_delta_gru/status.json`，完成后运行 `summarize_int_memory --root /data1/tzm/workspace/diagnostics/int_delta_instant_20261017 --modes delta_gru --output /data1/tzm/workspace/diagnostics/int_delta_instant_20261017/final_report` 并归档，再决定机制或执行方式是否值得继续修改。旧两个根目录的训练/消融均已收尾，不重复启动。

固定 256 帧 profile 已完成并归档：[profile_256.json](int_spike_results/delta_instant_20261017/profile_256.json)。模型墙钟中位数 37.77 ms，FM 均值 8.37 ms、backbone 21.44 ms、shrink 4.87 ms；该样本读出刷新位置 54.19%、8×8 刷新块 74.60%。仍是包含插桩的稠密实现，未实现或证明端到端加速。后续只等完整历史回放。

## 2026-10-17 即时版本消融完成，暂不追加 delta 训练

即时版本三项回放已全部完成，逐帧顺序、覆盖、权重哈希和 aligned AP 与最佳训练轮的一致性审计通过。证据：[final_report/report.json](int_spike_results/delta_instant_20261017/final_report/report.json)。

| 模型 | 对齐 AP70 | 清空历史 AP70 | 不对齐 AP70 | 对齐历史收益 |
|---|---:|---:|---:|---:|
| spike_gru | 75.9872% | 74.2929% | 71.5460% | +1.6944 pp |
| 原误差积累 delta_gru | 73.2605% | 72.7762% | 57.4270% | +0.4843 pp |
| 即时状态差，重新训练 | 74.6976% | 73.5844% | 66.5769% | +1.1132 pp |

即时版本恢复了部分历史收益，但精度仍低于 spike_gru，75.11% 的 8×8 块需要刷新且缓存为原来的 3 倍。目前不继续训练 delta 变种，也不直接投入其稀疏算子实现。保留它作为误差积累机制的探索记录，回到精度较好的 spike_gru 检查是否存在更合适的计算复用位置。

新增只读诊断 `opencood.tools.profile_int_readout_change`：在原 spike_gru 最佳权重的完整验证序列上，记录 FM 读出 Y 与位姿对齐上一帧读出的差异，统计 8×8 块的相对 RMS 差异超过 0/0.05/0.1/0.2 的比例；新视野和重置帧强制视为活跃。模型仍完整执行，预测不作任何替换。合成同值、单块变化及无历史覆盖检查先通过才执行流式诊断。

该诊断是每帧使用真实上一帧读出的单步统计，不模拟多帧陈旧缓存，不传播主干感受野邻域，也不证明 AP 或实际加速。用于判断是否值得在保留现有状态更新的前提下研究读出/主干的增量计算。

**当前唯一待跟进任务**：181 原 spike_gru 根目录 `/data1/tzm/workspace/diagnostics/int_spiking_gru_20261016/readout_change_20261017`，GPU 1，初始 PID 51436，代码 `d2268b6`；查看 `status.json`、`run.log`，完成后归档 `report.json`，如需核验顺序检查 `frames.jsonl`。启动记录 [readout_change_launch.json](int_spike_results/spiking_gru_20261016/readout_change_launch.json)。保持本次短后处理 30 分钟检查，禁止重复启动已完成的三个训练/消融系列；若无进一步值得运行的检查则结束该阶段监控。

## 2026-10-17 读出变化诊断完成，本轮监控结束

完整验证 5,794 帧、1,738 个标注帧、214 个序列重置已完成；逐帧 frame/segment/supervised 顺序、汇总统计及配置/清单/权重 SHA256 独立复核通过。报告与审计见 [readout_change/report.json](int_spike_results/spiking_gru_20261016/readout_change/report.json) 和 [audit.json](int_spike_results/spiking_gru_20261016/readout_change/audit.json)。原始逐帧证据保留在服务器 `readout_change_20261017/frames.jsonl`。

| 8×8 块相对 RMS 变化阈值 | 判为需更新的块比例 | 单步候选可复用比例 |
|---|---:|---:|
| 0 | 约 100.0000% | 约 0% |
| 0.05 | 99.9928% | 0.0072% |
| 0.10 | 99.9673% | 0.0327% |
| 0.20 | 96.1831% | 3.8169% |

结论仅针对当前 spike_gru 读出、8×8 分块和该相对误差度量：即便将阈值放宽到 0.20，单步复用候选也不足 4%，尚未计入主干卷积邻域与缓存维护成本，不值得据此实现稀疏算子或追加复用训练。这并不排除更粗粒度、任务相关选择或其他状态表示存在复用机会。该统计使用真实上一帧读出，没有多帧缓存误差、跳算 AP 或实测加速，不能将阈值 0.20 解释为可接受的检测误差。

阶段结论：同参数量 spike_gru 已提供精度可用性依据（本 seed 验证 AP70 75.9872%，GRU 75.8736%），但尚未证明计算优势；delta 的两次完整训练和本次读出诊断未提供值得继续工程化的精度/成本折中。本阶段收尾，不重复恢复平台期训练或继续扫描阈值。若继续新设计，优先考虑由检测任务约束的局部更新决策和显式更新预算，先做一个有真实缓存轨迹的最小验证，而非依据脉冲率推断可跳算比例。

**当前无本线程待运行的实验。** `int` 定时任务在本次归档同步完成后删除，结束本轮轮询；不影响其他线程的训练。保留全部模型、配置、日志和结果，研究总目标仍未完成，后续是新的机制设计阶段。
