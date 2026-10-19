# 当前主要路线：连续膜电位、余量与学习渐进发送的联合实现

## 用户最新要求与边界（2026-10-19）

用户明确希望三个部分一起实现并作为主要路线推进，不能只停在文档。总目标仍是SNN风格多车协作的合格通信量—AP；联合方案是当前优先实施路径，不是无论效果如何都必须保留的验收条件。Count3+Residual仍是有效对照，未启动的重复种子验证后移，不占用联合路径的优先资源。

## 已接入同一检测forward的首版

1. 车端捕获3D主干末端Count4之前的连续张量U0；初始化Q0=Count4(U0)、R0=U0-Q0。U0和R0实际进入接收计算，不再读取后丢弃。
2. 路端共享成熟SNN编码器，保留原L1选择25%体素。可训练无偏置P把128通道变为32通道Count3消息；连续小排序头仅看发送端32通道计数和局部均值上下文，学习将非空体素整行按分数分配给三个消息批次。所有硬消息仍为0..3整数，可用现有两位codec传输；软成员函数只作有偏替代梯度，不上网传输。
3. 每批先在三维源/目标网格与位姿下对齐32通道，再由可训练无偏置D映射回本地128通道语义。接收端使用由初始本地状态计算的固定帧内门控，把该批映射成增量电流I_k。门控是连续1×1卷积，初始权重0；尺度参数初始0.1。
4. 更新U_k=Q_(k-1)+R_(k-1)+I_k，读出Q_k=Count4(U_k)，保留R_k=U_k-Q_k。残余包含阈下、抑制及饱和信息；下批只加入新消息。Q是有界计数状态，相邻Q差可为负，不称全程正脉冲、纯AC或严格时间步LIF。
5. 每个Q_k经HeightCompression、冻结ANN BEV及连续检测头产生检测；同一训练forward对三个前缀的检测损失取均值。P、D、排序头和接收门控/尺度一起更新；初版成熟SNN编码器、ANN BEV/连续头及BN冻结。这是联合消息/融合训练，不声称整个原始编码器都已联合更新。

首版学习的是体素消息批次顺序，单体素32通道一次发送，后续批次补充其他位置；不是逐个脉冲的可学习发放时刻。不同源体素在三线性对齐后可能影响同一目标位置，此时余量按目标位置与通道保留。发送排序没有GT、接收端特征或位姿输入。

门控在同一帧各批固定，因此保持余量时完整电流具有可加性；完整结果不依赖排序的理论恒等性不是创新证据，排序的价值需要由早期前缀AP—字节证明。独立CUDA求和可能有舍入差，缓存直接/codec路径仍要逐值核验。完整单包运输对照在接收端重算相同排序再按相同批次接收，仅隔离包头开销；额外计算计入计时，不能作为优化单次推理时延。

## 空间、语义、生命周期契约

每帧状态绑定源/目标frame、实际时间戳、sender、位姿、三维网格、共享编码器SHA和P/D量化basis SHA。先对齐三维位置再通道展开和积分，HeightCompression在积分之后。不同帧/不同basis/变化位姿/错误批次拒绝；重复源体素行拒绝。空消息或无覆盖区域保持本地状态，不把缺包当空闲空间。只支持当前单路端、同帧内协作，不把DAIR车路证据写成已完成多车泛化。

## 已实现与验证状态

- 模块：`opencood/models/fuse_modules/e3dsnn_joint_membrane.py`。
- 检测器：`opencood/models/e3dsnn_joint_detector.py`，复用真实编码、点云数据、codec和检测头。
- 训练器沿用已有多前缀损失，新增明确joint状态/模块manifest；旧内容模型的运行语义不变。
- 本地18项检查通过，其中7项新检查覆盖本地阈前值的响应、跨批余量的因果效果、平移位置正确性、错误contract/重复包拒绝、空消息与帧隔离、抑制/饱和残余、内容/排序/接收模块共同获得梯度。
- 真实增强点云batch2审计入口：`python -m opencood.tools.audit_e3dsnn_content_detector --root /data1/tzm/workspace/diagnostics/e3dsnn_joint_audit_20261019 --family snn --learn --joint`。从已完成内容学习best移植P/D和编码器/后端；旧Residual参数保留但冻结且不进入新forward。
- 当前还没有该联合架构AP，不以单元检查代替真实点云、训练或精度证据。

## 当前下一步与预算

先完成真实点云审计：三个前缀direct/wire、完整单包同序列对照、重复forward/逐样本隔离、source ego等价、precount扰动对最终检测输出的影响、联合梯度和冻结校验、100%消息dropout身份路径。短审计可在显存充足时设置进程显存上限执行，不启动长期竞争训练，不终止其他任务。

通过后登记完整联合模型和分离的no-local/clear-interbatch/frozen-order消融。no-local只删除阈前本地信息，clear-interbatch保留初始本地余量但不把已接收批次余量传到下一批，避免旧实验同时删除两种余量的混淆。各组同初始化、10额外训练轮、原优化器/数据及第三前缀AP70选择（含epoch0）；逐帧实际字节、强ANN对照、真实codec及CPU复算要求保留。没有通过接口门槛不得开训；不得把准备状态写成训练中。

资源受限时优先联合实现/审计/首轮训练，其次必要机制消融；旧Residual重复种子可后续恢复。最终根据真实通信量—AP决定优化与取舍，不能为完成三项而忽视总体效果。每轮先读RESEARCH_GOALS顶部，再读本文件和最新证据；90分钟自动提示“循总目标推进联合，无变化静默”。


## 真实点云联合审计通过，首轮训练协议（2026-10-19）

本地19项检查通过；181增强点云batch2审计通过，代码29b1aa5。三前缀direct/真实codec预测逐值一致、完整单包重建同序列预测一致、重复forward无状态泄漏、批量/逐样本最大误差0；ego路径与成熟源模型精确一致，100%消息dropout三个前缀均回到ego。本地precount扰动实际影响检测输出，三项机制不是未使用的接口占位。

检测损失对P/D、发送排序、接收门控/尺度均有有限非零梯度，优化一步后内容basis变化；共享3D编码器、ANN BEV/连续头、BN摘要不变，审计优化步丢弃。总参数3120631，可训练25634；旧Residual参数保留但冻结、不进入该forward。原始[审计](results/joint_membrane_20261019/audit.json) SHA256 `4415f2f6d027130c2538e3f5ec354874e547e0da7bff111a8755a7fc46ae273c`。首次失败是旧codec不允许新批次元数据/两位分批，已添加隔离的joint_batch/joint_full协议类型并补往返检查，[失败日志](results/joint_membrane_20261019/first_protocol_failure.log)保留；没有失败训练。

审计两帧首包2966/3164字节、三包8988/9481字节、完整单包4780/5195字节，仅用于接口核验，不能当全量均值或AP—字节点。尚无新架构检测AP。

### 先执行完整联合模型，再做分离消融

当前优先只启动`joint`一组，需要一张空闲GPU，不等待四卡齐备。所有三个机制开启，来自内容学习SNN best（SHA `5abdd581b3d31ad11f32c66b290c99f129bda748a79e11d34214250426aa0902`）的3D/后端/P/D初始化；新门控与排序头按20260929固定初始化。初始联合模型SHA `107451f95c7785d8c67313f464694cc1706d20527164adf9dbe4f27b4aba4259`。

先全量零适配p1/p2/p3/完整单包及CPU复算；通过后10轮，batch2、Adam5e-5、原7/9里程碑、三前缀检测损失均值＋一次ego辅助项、消息dropout0.1，编码器/ANN BEV/连续头/BN冻结。按第三前缀AP70选best并包含epoch0；最终同一个best评估p1/p2/p3/off/full，1738帧/28622GT、真实包字节和CPU BEV/3D复算，独立full/p3容差不超过0.02个百分点。每次推理自身复算精确；训练上限6h，每项推理/复算上限1h，无自动失败重试。单组磁盘启动余量门槛6GiB。

以真实AP—通信判断该联合模型有无协作收益；目标超过固定ANN/SNN单车并保留强ANN/Residual对照，不要求刚初始化就达标。10轮后结项，不自动扩大超参数搜索；再按同初始化/预算登记no-local、clear-interbatch、frozen-order分离消融。首轮单组结果不能当三项各自有效的因果证明。

启动器`opencood/tools/launch_e3dsnn_joint_round.py`：从通过的`e3dsnn_joint_audit_v2_20261019`准备到`/data1/tzm/workspace/diagnostics/e3dsnn_joint_round1_20261019`。当前真实审计完成，训练尚未启动；按后续准备状态决定是否执行`--prepare`，禁止覆盖既有目录或重复launch。后台任务以本路线优先，旧Residual重复种子后移。


## 联合模型流水线已准备，等待一张空卡（2026-10-19）

181与本地各19项CPU检查通过，真实点云审计通过。`e3dsnn_joint_round1_20261019`已完成prepare，初始权重/五个配置/真实审计SHA核验通过，零适配4次推理＋4次CPU复算、适配10轮＋5次推理＋5次CPU复算的任务计划已检查。prepared审计SHA `bfe46ec148fae1c9e1cfb21821fe9bcb5b46e4ad4d57bc92fdb639aaf9885222`；[启动前检查](results/joint_membrane_20261019/prelaunch_check.json)、[配置审计](results/joint_membrane_20261019/prepared_audit.json)、[运行状态](results/joint_membrane_20261019/round_status.json)、[服务器CPU检查](results/joint_membrane_20261019/cpu_tests.txt)已归档。

当前八卡均有其他进程，不满足长期训练空卡门槛；状态为prepared，零适配全量推理与正式训练均尚未启动。此前真实点云短审计在GPU2限制20%进程显存预算、180秒上限完成，不代表已获整卡长期训练资源。后续每次优先检查此目录，获得一张空卡后直接运行`python -m opencood.tools.launch_e3dsnn_joint_round --root /data1/tzm/workspace/diagnostics/e3dsnn_joint_round1_20261019 --gpu <空卡>`；不用再次prepare、不等待四卡、也不优先启动旧Residual重复种子。若launch.json已存在先检查任务状态，不重复发起。保持90分钟跟进、无变化静默。


## 用户授权共享GPU，执行门槛更新（2026-10-19）

用户明确允许使用已有进程占用但利用率较低、显存充足的GPU，后续不再把“必须空卡”作为阻塞理由。联合路线优先级和实验协议不变，不停止其他进程。最近5次全卡采样中GPU3利用率9/25/0/8/8%，平均10%，可用16640MiB，明显低于其余卡，选择GPU3。

联合启动器新增显式`--allow-shared --memory-fraction 0.2`：每次准入采样5次（间隔2秒），平均利用率≤50%、每次空余显存≥max(8192MiB,20%整卡容量+2048MiB)。给PyTorch缓存分配器设置20%整卡上限，约4.7GiB；CUDA上下文和第三方分配不受该API完全约束，预留至少2GiB额外空间。准入只控制新阶段启动，不因本任务运行后利用率升高而终止正常工作。

原`e3dsnn_joint_round1_20261019`已prepare，不重建或覆盖。启动命令：`python -m opencood.tools.launch_e3dsnn_joint_round --root /data1/tzm/workspace/diagnostics/e3dsnn_joint_round1_20261019 --gpu 3 --allow-shared --memory-fraction 0.2`。零适配复核通过后自动进入10轮训练；阶段间若外部负载暂不合格，记录waiting_resources并等待，不重做已完成步骤、不重复launch。显存不足或真实任务错误按原规则记录失败，不自动重试。

代码已补低负载、显存不足、高负载、非法比例的准入检查，并保持旧调用默认空卡策略兼容。实际启动状态随后以launch.json、round_status和子进程日志为准。
