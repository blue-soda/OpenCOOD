# 消息压缩与完整模型差距的逐目标诊断（2026-10-18）

## Material Passport

来源：已全部复核的16点固定权重通信评估；用户授权按总目标推进。属于开发验证集上的事后描述分析，不是因果模块消融。目的：为下一轮SNN前端/融合任务适配选择明确问题，避免继续优化已不是主要差距的编码格式或BEV。

## 固定预算与配对

仅复用已有1738帧预测，无新增GPU推理、训练、阈值或检查点搜索。沿用diagnose_e3dsnn_ap70，逐帧核验哈希/GT并精确复现AP50/AP70，统计TP保留/丢失/新增、误检IoU分组、PR分段及丢失目标几何误差。单条CPU流水线四个子任务，各1h超时，失败不自动重试，四项完成即退出。

| 任务 | A（旧参数名ego） | B（旧参数名fusion） | 目的 |
|---|---|---|---|
| snn_compression | SNN编码器＋ANN接收端完整消息 | 同权重SNN保留25% | SNN压缩影响 |
| ann_compression | ANN完整FP32消息 | 同权重ANN2bit保留50% | ANN压缩影响 |
| full_gap | 完整SNN编码器＋ANN接收端 | 完整ANN | 压缩前差距 |
| lowrate_gap | SNN保留25% | ANN2bit保留50% | 约16kB差距 |

四项中的ego/fusion只是复用工具A/B参数名，不能按车端关闭/开启解释。双方都使用协同输入及ANN接收端，来源协同适配预算均40轮。后两项既改变编码器也改变共同适配的接收端权重，不能把差距单独归因编码器量化。丢失目标几何统计是按结局选择的子集，不代表总体因果效应。AP依赖排序与一对一匹配，不以TP数量替代AP。

独立根目录/data1/tzm/workspace/diagnostics/e3dsnn_rate_diagnostics_20261018，计划见results/rate_diagnostics_20261018/plan.json。复用已审阅的supervise.py，CPU环境CUDA_VISIBLE_DEVICES为空。

下一步研究方向：固定成熟ANN BEV及连续检测头，把训练资源用于SNN前端与融合在通信预算下的任务适配；必要时加入ANN教师的检测监督，但必须设置不含教师的同预算对照和量化ANN参照。教师蒸馏、消息可用性建模是否有用均待实验，不能依据本次描述统计直接宣称。正式训练协议在本诊断复核后单独登记，不预先启动无边界长训练。

## 实际启动与续接

四项CPU诊断已于2026-10-18启动，监督PID39525，首项snn_compression子进程39527；CUDA_VISIBLE_DEVICES为空，不占用GPU。运行提交3886b90aa638fe8af3b867f422f4c78f0088f90f，计划SHA256 7855906a1fc358160070e2bc1b10d440ee21696fcd5c9f766675e4b4de99fa04。复用监督脚本SHA256 3a661c319d8dde0949d14dff1e9134199f6993d426d4d3f9f23c0576b4b2bd84，每项有1h超时，失败停止后继项、不自动重试。

证据：[launch.json](results/rate_diagnostics_20261018/launch.json)，SHA256 63401d6457f566ee229b7ba0d6a2fee4d9c27a32b79165fd83e04610339dba9e；[startup_pipeline.json](results/rate_diagnostics_20261018/startup_pipeline.json)，SHA256 4de8a0d6c87c905b0caf88732c58f82fcfc7802ab1859109a969bf7bfb46e090。

下一次检查远端rate_diagnostics_pipeline.json；complete后依次取回四个同名结果JSON，核对passed/精确AP及A/B目录对应关系，再按总目标登记有界训练方案。running则检查日志推进，不再次启动。自动任务继续90分钟，无变化静默。

## 最终核验与后继（2026-10-18）

四项已完成，均1738帧、28622GT，A/B逐帧GT和预测哈希一致性通过，精确重现归档AP50/AP70。

| 对比A→B | 保留TP70 | 丢失 | 新增 | 共同未检 | AP70变化(pp) |
|---|---:|---:|---:|---:|---:|
| SNN完整→保留25% | 18424 | 386 | 188 | 9624 | −0.45197 |
| ANN完整→2bit保留50% | 18621 | 376 | 200 | 9425 | −0.36409 |
| 完整SNN→完整ANN | 17694 | 1116 | 1303 | 8509 | +1.00824 |
| 低通信SNN→低通信ANN | 17486 | 1126 | 1335 | 8675 | +1.09612 |

SNN与ANN压缩分别净少检198与176个目标，FP同时减少427与362。SNN压缩在recall>0.5段的AP贡献损失0.50618pp，ANN为0.43450pp，低召回段略有改善抵消部分损失。完整模型ANN在recall>0.5段的AP贡献高约0.72883pp；低通信时高约0.80052pp。差距集中在高召回尾部的描述，不能直接推出校准、NMS、量化或某层的因果责任。

结束描述诊断，不继续扩充分析项。下一轮固定各自成熟ANN BEV及连续头，用两类编码器×融合only/前端joint的匹配10轮实验检验消息条件下的任务适配，见[BUDGET_ADAPT_20261018.md](BUDGET_ADAPT_20261018.md)。先保留ANN融合以隔离前端因素，不能写成已改进SNN融合。

原始证据目录results/rate_diagnostics_20261018：snn_compression.json SHA256 b65959fa2366aa40b7c5751ff1112034461e6a4b4136511b4485947826a14ac3；ann_compression.json a448ddf2c538419152d0857dac08a99969b1de725bd9b8205bbf153bbf716ec4；full_gap.json c916d888eb3aa4d14b3d6f3813dfa27ffdea2f5e517467ebc777a181c6949572；lowrate_gap.json cda17e603080031eb3d780430d5863591fd0699f54397108f7a79a8437d90bc2；完成状态7c440b8fce9cec0e2b6effbfb7ba6de4d4509508219ef710b1b27749cff3ccc7。远端原始字节已核对并镜像。
