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
