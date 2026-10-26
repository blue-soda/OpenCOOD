# InfoCom 1%工作点适配协议（2026-10-26）

本阶段为方案二的强ANN对照补齐训练机会；不改变SNN两臂、不恢复方案三。实现后先做两次丢弃权重的真实优化步检查，正式训练须另通过GPU/存储准入。本协议先于任何正式适配AP登记。

## 唯一主实验

从已复评官方epoch7权重开始，固定1%空间mask，沿用官方评分、4bit STE、train时mu+eps*std与eval时mu。使用与SNN同一manifest的4678有效训练ID、1738共同验证ID，全部有效参与方；保留InfoCom本身的体素化/标签/网络，最终用共同GT重算，不能称架构相同的激活消融。

10个适配epoch、batch2、每轮2339步、seed20260929，Adam lr=5e-5、weight_decay=1e-4、eps=1e-10；完成第7/9轮后lr乘0.3，梯度裁剪10。训练所有发布权重，sigma→std兼容映射保留，新增identity AdaptiveScale固定；所有BN running buffers冻结并逐轮核验，BN仿射参数允许训练。训练范围比SNN的局部3D解冻更宽，给予ANN充分任务适配机会，不能称相同可训练参数量/计算成本。

目标为检测损失 + 1e-4 KL，KL从本次适配第1轮开始；beta取官方配置，不搜索。**这是预训练模型的低比例适配，不是精确续跑作者从头训练日程**：作者原代码在epoch<=15不加KL，本实验显式提前使用现有IB正则，保留随机潜变量。没有加入SNN的额外ego分支、蒸馏或新接收模块。

每轮在1%点做真实wire推理、逐包SHA/预测SHA/身份检查和共同GT CPU AP复算；以1% AP70选择10轮中的最佳，平分取最早。训练RNG和单独验证进程隔离。选择点是空间比例，**不是每帧2KiB硬预算**；所有结果报告均值/P95/最大真实bytes，不依靠学习后更大的包量制造同预算优势。最终最佳再做1%/2.5%/5%/10%固定曲线，保留未适配曲线。

## 检查点、恢复和资源

每轮last.pt原子保存当前model、Adam、Python/NumPy/Torch/CUDA RNG、协议、历史、最佳模型及其epoch/AP。完整最佳权重纳入同一个恢复事务，避免“best已覆盖、last未写完”造成选择记录错配。恢复只从已完成轮数继续；未完成验证使用新的attempt目录，保留失败证据。满10轮导出best；最终四比例评估仍是单独必需收尾，不能把trained状态当作研究完成。

正式启动要求至少6GiB可用存储，并在每次原子写入前检查预计临时文件＋512MiB余量。完整模型约0.48GB；含最佳权重的Adam恢复对象约1.92GB，原子替换时还需并存旧last和临时文件，另有候选模型、最佳导出及逐帧预测。训练GPU需满足低利用率且至少18GiB空闲；实际审计显存与Adam后显存另记录。任何数据、非有限值、wire、磁盘或子进程失败均停止并留证，不自动改batch、减轮数、跳帧或换种子。

当前181 /data2仅约2GB空闲，146新CUDA进程仍无法初始化；187有54GB磁盘但每卡仅1674MiB空闲，无法承载已测真实批次。不得终止其他用户任务或删除其他数据腾空间。正式训练未启动。

## 退出与解释

只跑这一组10轮，不看中途AP追加学习率/比例/epoch搜索。最终按真实字节对比Scheme2固定/多预算和原ANN；若低预算适配ANN反超，记录反超，不隐藏强对照。SNN是否有优势仍需统一计算测量和匹配训练的内部ANN对照，不能只靠外部模型低比例曲线归因SNN机制。后续计算/机制消融优先，停止重复冻结比例探索。


## 2026-10-26 05:59触发：训练器准备完成，存储准入未满足

实现cf79ce0、22162c6已完成固定1%适配训练器、严格来源加载、源代码快照、每轮真实wire/共同GT复算、原子可恢复检查点、最佳权重选择及末尾四比例复核。末尾只有全部复核通过才标complete；trained是中间状态。启动前检查6GiB磁盘及五次GPU准入，172800秒外层上限的具体启动计划已保存，当前prepared_not_launched。没有新增正式训练AP。

181 GPU4完成两次有界真实Adam烟雾检查（第二次验证最终快照/GPU准入改动）：每次两步/四个训练样本，保留随机latent，Ldet+1e-4 KL；两次loss逐值一致1.4971104860、1.4383445978。第2步KL=486.71698；BN统计逐值不变，权重全部丢弃，未作为正式训练warm start。峰值allocated=5,601,183,232 B，约5.60GB/5.22GiB，较之前纯梯度检查增加Adam状态。不能将两步loss变化解释为验证AP收益。

三项恢复测试在本机和181均通过：Adam和四类RNG恢复后下一步参数逐值相同；磁盘不足不改已有last；模拟临时写入中断仍保留原last。真实大模型完整epoch保存/恢复及四比例收尾尚未实际运行，不能把小模型事务测试称为完成了正式训练复核。

当前资源：181 /data2剩2,085,552,128 B，低于6,442,450,944 B准入；146以正常枚举的GPU4 UUID新建进程仍CUDA unavailable；187虽有54GB磁盘，每卡仅1674MiB空闲，低于真实批次所需。正式训练仍未启动，需让181可用空间达到至少6GiB，或提供可用于此训练的同等级存储/算力位置。未删除他人文件，未reset GPU。现有方案二固定组第8轮601/2339、多预算第1轮901/2339继续，最佳仍58.85365%，没有新最终AP。

[最终两步检查](results/strong_ann_baselines_20261025/infocom_adaptation_ready_20261026/infocom_adaptation_smoke02/smoke.json)、[具体启动计划](results/strong_ann_baselines_20261025/infocom_adaptation_ready_20261026/infocom_lowrate_launch_plan.json)、[测试与归档SHA](results/strong_ann_baselines_20261025/infocom_adaptation_ready_20261026/verification.json)。后续先检查现有输出目录和PID避免重复；空间恢复后重新GPU准入，再按计划启动同一个10轮实验，不重新选择超参数。


## 2026-10-26 12:16触发：磁盘恢复，正式训练已启动

181 /data2实测可用161,176,248,320 B（约150GiB），此前6GiB存储准入阻塞已解除，无需等待额外许可，也不再重复询问空间。未由本任务删除其他用户数据，未推断是哪项外部操作释放空间。

按原先已登记的唯一1%工作点协议启动infocom_lowrate_adapt01，超时父PID6042，181 GPU6；选择空闲卡避开GPU0多预算SNN和GPU4四预算收尾。GPU6五次准入均24252MiB空闲/0%利用率，训练器内部再次检查。启动前确认输出与启动标记均不存在、/proc没有另一个正式适配进程，并核对当前训练脚本SHA等于第二次真实Adam烟雾检查的版本859fce7d3ae0189b41ffaad66806a7c0af51c37e32a9f68537e6a253242a8659。只有GPU选择由原计划4调整为6，模型、数据、种子、loss、10轮和选择规则均未改变。

正式首步loss=1.2924560、task=1.2207047、KL=717.5141、裁剪前梯度范数16.3888；第100步已处理200个样本，loss=1.0799453、task=1.0799142、KL=0.31105、梯度11.2834，均有限。全训练清单打乱后的首批不同于4样本烟雾检查，不能用二者首步loss不同判断恢复失败。这不是从烟雾权重续训，正式实验重新加载同一官方源权重。无验证AP或优势结论。

根目录/data2/tzm/cp_snn_recovery_20261026/infocom_lowrate_adapt01；外部启动标记为同级infocom_lowrate_adapt01.launch.json，原infocom_lowrate_launch_plan.json保留为历史准备状态。轮询时读取progress.json、failure.json及PID；status.json在正式训练结束/四比例收尾时生成，训练中缺少status.json本身不是失败。每轮保存完整可恢复last、验证和共同GT复算；十轮后四比例完整wire收尾，完成前不把训练损失当研究成果。

SNN多预算第2轮451/2339批，固定组8KiB评估继续。old_snn的0/2/3/4/6/8KiB预测均已生成1738帧，正在执行原登记full点（本轮见4帧），尚未完成总汇总；不重复启动旧实验或提前宣布最终通过。

证据：[正式启动与资源准入](results/strong_ann_baselines_20261025/infocom_adaptation_launch_20261026/infocom_lowrate_adapt01.launch.json)、[实际训练协议及脚本快照SHA](results/strong_ann_baselines_20261025/infocom_adaptation_launch_20261026/infocom_lowrate_adapt01/protocol.json)、[首轮进度](results/strong_ann_baselines_20261025/infocom_adaptation_launch_20261026/infocom_lowrate_adapt01/progress.json)。


## 2026-10-26 13:15触发：InfoCom首轮适配已完成共同GT复核

正式1%适配完成第1个epoch，已进入第2轮（本轮读取1100/2339步）。共同1738帧、28622个GT下AP30/50/70为79.74066% / 74.93195% / **60.93306%**；采用共同GT CPU结果，而不是官方GT的60.95541%。官方与共同GT存在历史记录的坐标差异，不能混用两套AP。

真实通信包含80 B封装，均值**2043.013809 B**、中位2044 B、P95 2205 B、最大2258 B。1738帧中843帧超过2048 B；1%是空间选择比例，不是逐帧2KiB硬上限。包与预测SHA、身份、载荷解码和形状已重新全量核验，独立CPU复算验证1738份预测及共同GT摘要；本机取回后再次验证索引SHA与逐帧字节均值。该epoch权重SHA为418d1d11c78de683fa72e658f0d218e9d38b84527f7a9912619498b2f8d2f4e8。

| 工作点 | 状态 | 共同GT AP70 (%) | 平均B/帧 | P95 B | 最大B |
|---|---|---:|---:|---:|---:|
| InfoCom官方权重，冻结1% | 完成复核 | 56.88354 | 2176.936 | 2341 | 2411 |
| InfoCom 1%适配，第1轮 | 中途验证，非最终模型 | 60.93306 | 2043.014 | 2205 | 2258 |
| 方案二固定2KiB联合训练，最佳第6轮 | 10轮完成，2KiB点复核通过 | 58.85365 | 2046.886 | 2048 | 2048 |

首轮适配InfoCom较自身冻结1%提高4.04952个百分点，平均通信下降约6.15%；相对方案二当前2KiB点，AP70高2.07940个百分点且平均字节略低。因此**按平均通信量—AP，此时不能宣称方案二优于适配ANN**；硬上限不同也不等于SNN已经取得可发表优势。保留此强对照，继续双方已登记训练及完整曲线。InfoCom更广的训练范围、不同骨干和训练起点仍是跨架构比较边界，不把差距归因为神经元本身。

本轮不追加比例、学习率或新接收模块，不重启方案三。先完成InfoCom原10轮及最终四比例复核、方案二多预算10轮、固定组8KiB收尾；之后按完整结果选择匹配码率/计算测量和机制消融。8KiB日志已到500/1738帧，多预算SNN第2轮751/2339批；未把未完成评估记为结果。尚无统一计算优势结论。

证据：[重新核验摘要](results/strong_ann_baselines_20261025/infocom_adaptation_epoch01_20261026/verification.json)、[训练历史](results/strong_ann_baselines_20261025/infocom_adaptation_epoch01_20261026/history.json)、[逐帧字节](results/strong_ann_baselines_20261025/infocom_adaptation_epoch01_20261026/epoch01/per_frame_bytes.csv)、[共同GT复算（gzip）](results/strong_ann_baselines_20261025/infocom_adaptation_epoch01_20261026/epoch01/common_gt_replay.json.gz)。原始全部预测和真实包保留在181验证目录；压缩索引解压后SHA与审计记录一致。
