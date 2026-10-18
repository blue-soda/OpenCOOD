# 统一BEV状态与局部更新路线

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan / run
- Origin Date: 2026-10-18（Asia/Shanghai）
- Verification Status: IMPLEMENTED，待服务器工程与训练验收
- Version Label: unified_bev_dense_v1

## 目标与现有证据
目标是在检测精度尽量保持的前提下，使昂贵计算集中在变化区域，而非只降低理论MAC或脉冲率。用户已授权改变ResNet+多尺度解码结构，先整图更新基线，再加入区域选择和真实局部执行。

保留对照为原ANN GRU（完整AP70=75.873610%，模型平均32.2585ms）及period3独立二/三级修正（75.823490%，26.8356ms，平均下降16.81%）。这是同GPU顺序单轮模型时间，不含IO/体素化/后处理。此前原ResNet局部裁剪变慢，小修正模块整图执行优于局部，故不能预设稀疏一定更快。

解码后特征修正仍在181 GPU7训练（decoded_foreground_period3_resume_v2）；它保留冻结原检测器、只学41,984参数补差，和本次重新学习状态表示及检测头不同。不取消其健康训练，不以它的低精度证明新路线失败。

## 首个可训练模型
输入仍是相同DAIR同步自车点云序列；继承冻结教师Pillar编码器和scatter，去掉原GRU、三个ResNet阶段、多尺度解码及shrink。学生模型不包含这些模块，不用教师缓存初始化任何学生状态。

1. 当前64×200×504 Pillar图，3×3 stride2卷积到96×100×252，再逐通道3×3和1×1，产生便宜的当前观测。
2. 观测平均池化4倍，1×1到32通道、逐通道5×5，再最近邻上采样，提供低分辨率上下文。首版每帧算该分支；低频上下文缓存尚未实现。
3. 唯一持久状态128×100×252，使用现有SE(2)最近邻warp对齐上一帧状态。输入为观测96+上下文32+历史状态128+有效覆盖1。
4. 局部更新为1×1混合、两个逐通道3×3（中间1×1）、1×1输出候选修正及保留门。候选还接收当前证据的1×1投影。保留门乘valid，新视野不依赖无效历史。更新的空间半径为2，无跨空间归一化/全局注意力；未来局部裁剪针对这个模块，必须包含两格halo。
5. 1×1将状态读出为256通道，再用可训练分类/回归/方向头。检测头从教师对应头初始化，但读出和状态从随机权重学习；没有强制拟合教师解码后特征。

首版**全部位置每帧更新并写回**，全序列递归，不周期注入教师或原ResNet刷新。序列/显式重置/时间间隔>0.25s清空，重复或倒序帧拒绝。首版不训练运动预测、变化选择器、不进行区域跳算，不把小网络整体变快称为稀疏收益。新状态相比GRU语义和参数量均不同，不是同参数SNN消融。

## 训练与判定
- 教师：既有full_gru/epoch_017/trained.pth；setup核对数据清单、配置与教师SHA。数据8593训练扫描、5794验证扫描（1738验证标签），无随机延迟、无数据增强，不启动异步协作训练。
- 学生Pillar冻结/eval；观测、上下文、状态更新、读出、检测头联合训练。教师独立时序流，仅在训练提供输出蒸馏，不用于推理。
- TBPTT4真实连续帧，窗口内固定参数，每窗口反向一次；学生状态写回，窗口末detach但不清空；无标签帧仍参与递归和输出蒸馏。
- 损失=窗口内有标签帧GT检测损失均值 + 0.2×全部帧前景归一输出蒸馏均值。GT与KD分别归一，不让标签稀疏直接稀释GT。KD沿用前景框/分类/方向项；不使用教师特征MSE限制新状态语义。
- seed303，AdamW lr1e-3/weight_decay1e-4，梯度裁剪5；每轮完整验证；ReduceLROnPlateau factor0.5、patience2、绝对阈值0.001、最小lr1e-5。至少5轮，连续5轮无显著改善且lr<=1e-4停止；上限40不是收敛证据。保存每轮不可变student.pth，以及完整优化器/调度器/状态计数的latest.pth，可在新目录连续恢复。
- 先看完整AP是否达到工程可用区间，初步目标距原GRU不超过0.5个百分点；达不到时诊断容量/监督/时序漂移，不以低精度的速度作为成功。通过后同GPU顺序重载比较原模型、已有保留方案、新整图基线，报告平均/P50/P95及状态量。
- 后续选择器只看当前便宜观测、历史观测/状态、覆盖与年龄；训练集决定策略，验证集评价。变化块和新视野执行局部更新，不变区保留对齐状态；大量变化回退整图。需要对同一个网络比较整图/局部执行，计入warp、选块、halo、调度及头的开销。

## 工程门槛与文件
新增int_unified_bev.py与run_int_unified_bev.py。单测验证局部裁剪输出与同输入整图更新对应区域一致、未选位置不变、跨帧梯度、detach、场景重置与时间因果性。真实8帧检查输出shape/有限值、独立学生状态、原ResNet/解码/shrink零调用。新模型不应与教师预测相等，不伪造等价验收。

运行入口：`python -m opencood.tools.run_int_unified_bev --root /data1/tzm/workspace/diagnostics/int_spiking_gru_20261016 --output <新目录> --gpu <空闲GPU> --epochs 40`。
独立验证使用同入口加`--checkpoint <epoch/student.pth>`；续训使用`--resume <旧目录/latest.pth> --epochs <更大总上限>`，要求源作业完整结束且只是到轮数上限。

记录launch/provenance/training_contract/engineering/status/history/best、逐训练和验证帧以及每轮权重。监控检查PID和日志推进；不因耗时波动杀死健康训练，训练期间不发布正式速度比较。继续本地main编辑/push、服务器pull，保留其他研究的文件。既有路线级自动任务保持，首次启动验收后依据实测每轮耗时调整间隔。

## 2026-10-18 首次启动已验收
实现6f1e7e0已本地main提交/push并由181 pull；64项INT测试通过（新增裁剪同输入一致性、未选区不变、跨帧梯度与因果重置）。真实8帧验收PASSED，状态为1×128×100×252，原ResNet/多尺度解码/shrink调用均0，持久状态12,902,400字节。新模型209,396个可训练参数，冻结Pillar768参数；架构与原30M检测器不同，不作同参数对照。
当前活动实验unified_bev_dense_v1，181 GPU0 PID16151，目录/data1/tzm/workspace/diagnostics/int_cache_repair_20261017/unified_bev_dense_v1，已进入epoch1训练600/8593扫描、290标注、161次TBPTT更新；loss/梯度有限。尚未产生完整验证AP，不以训练loss或模块数量宣称精度或加速。启动/工程/契约归档docs/int_spike_results/cache_repair_20261017/unified_bev_dense_v1_*。Verification Status更新为ENGINEERING_VERIFIED / TRAINING_IN_PROGRESS。
下一轮检查本目录status/history/best和PID，重点核验首次完整5794/1738验证及学生状态自主递归；训练期间所有速度只作诊断，正式比较需完成后独立重载。旧decoded_foreground_period3_resume_v2在GPU7第31轮验证，最佳epoch30 AP70=66.68412%，仍单独按平台规则完成，勿混用两条路线checkpoint/训练契约。现有75.82349%候选保留。
当前路线级自动任务仍为90分钟（推进时序感知，仅完成异常通知），覆盖长训练；首个完整epoch后按实际每轮耗时及平台接近程度调整。无参数更新或日志长期不推进需查进程，不能单凭GPU利用率低判定失败。
