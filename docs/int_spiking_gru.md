# GRU 参数与缓存匹配的混合 SNN 原型

## Material Passport

- 任务：2026-10-15 用户要求构造与当前 GRU 架构、参数量匹配的 SNN 记忆模块。
- 范围：实现 `spike_gru`，通过两轮工程验收后开展配对全量训练；尚无精度改善结论。
- 数据：既有 DAIR 自车完整片段；与第一轮相同的冻结清单及 P1 初始化，无异步融合。
- 状态：181 上 GRU/spike_gru 正式训练分别在 27/21 轮达到预设平台期；训练完成审计通过，最佳权重历史消融回放进行中。当前执行信息以文末最新记录为准。

## 匹配范围

严格复用原 INT GRU 的层定义与参数名称：reset/update/candidate 均为 128→64 的 3×3 Conv+BN，前两者使用 Sigmoid，candidate 使用 Tanh；读出为 66→64 的 3×3 Conv+BN+ReLU。64 通道膜电位取代原隐藏特征，另保留原有 age 通道，无额外 trace、可学习阈值或时间常数。

记忆模块共 259,712 个参数，整个模型共 30,776,212 个参数。200×504 BEV 上 FP32 缓存为 65×200×504×4 = 26,208,000 字节，均与当前 GRU 完全相同。继承同一组空间网络、同一位姿 warp/重置逻辑、同一检测损失与训练器；相同 RNG seed 下各层初始化也相同。

匹配的是有参数的层、通道、初始化和缓存预算，**不是完全相同的计算图或 FLOPs**。门控、候选卷积及检测器保持 ANN；新增的是发放、复位和代理梯度。这是混合 SNN，并非全脉冲 GRU 或事件驱动 GPU 稀疏算子。

## 状态更新

输入 X，位姿对齐后的上一帧膜电位 U，门控与候选仍沿用原 GRU：

```
r = sigmoid(BN(Conv_r([X, U])))
z = sigmoid(BN(Conv_z([X, U])))
c = tanh(BN(Conv_c([X, r * U])))
V = z * U + (1 - z) * c
S = H(V / theta - 1) - H(-V / theta - 1)
U_next = V - theta * S
Y = original_readout([theta * S, age, occupancy_mask])
```

S∈{-1,0,+1} 是正、负两种二值脉冲的紧凑表示，每个真实扫描一次更新。保留双极性是因为原候选使用 Tanh，避免直接丢弃负值信息。theta=0.25 是固定超参数，在专用 YAML 中记录，不增加参数；由于候选有界，不使用会导致难以发放的默认阈值 1。选择 0.25 是工程起点，未按验证 AP 搜索。

读出使用脉冲，不能直接读取 U 或 V。原 GRU 已有的 age 与 occupancy_mask 两个辅助通道保持不变，因此也不能声称检测输出完全由脉冲决定。复位可微，二值阶跃使用既有 atan 代理梯度。连续 U 可以被门控读取，缓存仍为 FP32，不宣称存储压缩或节能。

## 验证与运行

语义测试 `opencood/test/test_int_spiking_gru.py` 覆盖：参数/层/初始化逐项匹配，双极脉冲，亚阈值积累与复位，无膜电位读出，跨帧代理梯度及 detach 边界，位姿对齐/场景重置/状态不变性和阈值合法性。同时重跑原 INT 上游等价性和 LIF 测试，防止基础状态接口回归。

配置：`opencood/hypes_yaml/dair-v2x/snn/int_ego_spiking_gru.yaml`；训练入口 `run_int_epochs --mode spike_gru --tbptt-steps 4 --fm-only-epochs 1`。原配置和历史实验不改动。真实数据试跑与新 GRU 对照使用同一新 YAML、同一 P1 初始化、seed303、相同两段 train/val，跑两轮以覆盖 FM 预热和联合训练；试跑 AP 不进入科学比较。用户随后指定使用 181 训练，正式训练如下。

## 2026-10-16 早期实施记录（184 阻塞，已由下节接续）

代码 `17af515` 已经本地 main push、服务器 pull。服务器使用 `python -m unittest discover -s opencood/test -p 'test_int*.py'`，27 项全部通过。必须用 discover：仓库另有 `opencood/test.py`，按 `opencood.test.<模块>` 导入会错误触发旧数据脚本。

实际分别构建完整 GRU/spike_gru，设相同 seed303 并加载原 P1，检查两者所有 state_dict key 及初始化张量均精确相等；实际参数计数均为整模型 30,776,212、FM 259,712。证据见 [implementation_audit.json](int_spike_results/spiking_gru_20261016/implementation_audit.json)。这验证的是初始化一致，发放/复位启用后的前向结果不应与 ANN 相等。

八张 GPU 当前均被现有 vLLM worker 占用，每张剩余约 1.6 GiB，不足以试跑本模型。没有启动试跑、没有中止其他进程或修改共享环境。恢复 60 分钟定时跟进，提示词“推进脉冲GRU，仅完成异常通知”，仅在 GPU 可用后启动下列既定工程验证。不要将尚未发生的试跑写成完成。

后续执行顺序：

1. 检查 `/data0/chen/gzc/workspace/diagnostics/int_spiking_gru_20261016` 是否已有 launch、pilot 输出或活动进程，避免重复。重新查询空闲显存与利用率。
2. 在服务器仓库用原 opencood Python 环境运行 `opencood.tools.launch_int_memory --stage pilot --modes gru spike_gru --gpus <两张实际空闲卡> --root /data0/chen/gzc/workspace/diagnostics/int_spiking_gru_20261016 --config opencood/hypes_yaml/dair-v2x/snn/int_ego_spiking_gru.yaml --manifest /data0/chen/gzc/workspace/diagnostics/int_ego_20260929/manifests_v3_decoded/sequence_manifest.json --checkpoint /data0/chen/gzc/workspace/logs/logs/dairv2x/repro_dair_single_pointpillar_wide_for_codyntrust_cobevflow_2026_09_11_02_51_02/net_epoch_bestval_at57.pth`。设置 `PYTHONPATH=.` 以及 OMP/MKL/OPENBLAS 线程数为 1。启动记录为 `launch_pilot_gru_spike_gru.json`。
3. 核验两轮完整 train/val 顺序及覆盖、有限 loss/梯度、FM 首轮空间权重冻结、第二轮空间权重改变；核验 spike_gru 正负发放率、状态大小与膜电位有限性，避免全静默/饱和。汇总观察到的实际结果，保留失败证据。
4. 完成原型工程验收后报告，不将小样本 AP 作为精度结论，不自动扩大为全量训练。未通过自身工程验收之前，不复用旧 LIF 的 engineering_gate.json 启动全量 spike_gru。

## 2026-10-16 181 正式训练（当前接续入口）

用户指定优先 181、故障时用 187 后，已迁移原冻结清单和 P1。181 的六处 DAIR 绝对符号链接仍指向 184，已仅对目标存在的悬空链接改为本机相对链接，保留修复审计；数据内容、清单顺序、位姿、时间和监督标志未改变。完整训练 8,593 扫描/4,678 标注，验证 5,794 扫描/1,738 标注，路径重定位后全数校验通过。187 未使用，也未修复其链接。

27 项 INT 测试在 181 通过。GRU/spike_gru 两轮试跑均通过覆盖、有限梯度、首轮空间冻结/次轮联合更新检查；spike_gru 第 1/2 轮训练非零发放率为 8.23%/9.17%，验证为 8.16%/8.94%。重启试跑与连续训练的模型、优化器、调度器、契约和 AP 精确一致。小样本 AP70 为 0，不作为准确率证据。

正式训练从原始 P1 重新初始化，启动代码 `125e3f9`，GPU 0 为 GRU（初始 PID 59597），GPU 1 为 spike_gru（初始 PID 59598）。两组同 seed303、AdamW 1e-4、TBPTT4、首轮只训练 FM、之后联合训练、冻结 BN、无增强，无人工延迟或异步融合。至少 20 轮，连续 10 轮无超过 0.001 AP 的改善且学习率降到初值 1/8 以下时停止，上限 40 轮；到上限而未达到平台期不能直接称收敛。

```text
host: mindspore-181
repo: /data1/tzm/workspace/OpenCOOD
python: /data1/tzm/miniconda3/envs/opencood/bin/python
root: /data1/tzm/workspace/diagnostics/int_spiking_gru_20261016
runtime config: <root>/inputs/config.yaml
manifest: <root>/inputs/sequence_manifest.json
P1: <root>/source/net_epoch_bestval_at57.pth
launch: <root>/launch_full_gru_spike_gru.json
outputs: <root>/full_gru, <root>/full_spike_gru
logs: <root>/full_gru.log, <root>/full_spike_gru.log
```

启动与审计原始证据存于 [spiking_gru_20261016](int_spike_results/spiking_gru_20261016/launch_full_gru_spike_gru.json)，同目录含 engineering_gate、relocation_audit、dataset_symlink_repair 和 pilot 启动记录。配置仅重定位数据路径；实际执行参数与无异步声明见两组输出下的 `effective_runtime.json`。

后续每 60 分钟检查这两个输出的活动进程、日志、failure/result/curve 和磁盘；启动时 `/data1` 剩余约 28 GiB，训练器在每轮开始前检查 5 GiB 下限。不要重复启动。仅在进程中断且未达平台期时，使用原环境运行 `opencood.tools.resume_int_memory --root <root> --launch-record launch_full_gru_spike_gru.json`，先查明中断原因并确认 GPU 空闲。两组完成后对最佳权重做 aligned/reset/no-align 冻结回放；严格比较本轮配对 GRU，旧结果保留为历史参照。只在完成或异常时通知。

## 2026-10-16 22:35 训练完成与消融接续

两组正常结束，无 failure 文件，均满足至少 20 轮、连续 10 轮无显著改善、学习率不高于初始值 1/8 的停止规则。核验 latest.pth 中的 epoch/stale/学习率与结果记录相符，最佳权重哈希正确；48 个 epoch 的训练和验证覆盖全部符合冻结清单，loss/AP 均有限。除 mode 外两组训练契约完全一致。证据：[training_completion.json](int_spike_results/spiking_gru_20261016/training_completion.json)。

| 模型 | 完成 epoch | 最佳 epoch | 最佳验证 BEV AP70 | 最终学习率 |
|---|---:|---:|---:|---:|
| GRU | 27 | 17 | 75.8736% | 3.125e-6 |
| spike_gru | 21 | 11 | 75.9872% | 6.25e-6 |

脉冲 GRU 比配对 GRU 高 0.1136 个百分点。这是单 seed、验证集选择最佳权重的结果，不足以证明稳定或统计显著优势。两者最后一轮 AP 均低于最佳轮，应使用冻结的最佳权重进行后续比较。

已在 GPU 0/1 分别启动 `replay_gru`（worker PID 4599）和 `replay_spike_gru`（worker PID 4608），顺序执行 aligned/reset/no-align，每项完整覆盖 5,794 扫描与 1,738 标注。下一次跟进检查这两个目录的 `status.json`、`process.json` 和各策略日志；不得重新训练或重复创建回放目录。回放器要求 aligned AP 与训练最佳轮相差不超过 1e-6，并验证权重哈希和逐帧顺序。待两组回放全部完成，再用 `summarize_int_memory --modes gru spike_gru` 生成配对报告；本次尚无清空历史或不对齐结果。磁盘约剩 20.76 GiB，定时任务继续有效。
