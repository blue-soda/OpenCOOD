# 联合通信目标候选：一次配对微调种子复核（2026-10-20）

## 启动前登记

目标：prefix_rate在首个种子中相对匹配control首包AP+0.14150个百分点、真实字节-14.08%，先检验是否依赖一次数据顺序/dropout，而不是立即扫系数。三项联合机制保留；本轮不改变架构、稀疏代理、权重或后处理。

仅两组control、prefix_rate；训练seed改为20260930，其他设定与前轮同组完全一致。两组都从原联合第7轮best `f4a5795b8b001a7cc4c04fd9eb3959f02df70181afe23691e733cb3aa4eb8588`开始，绝不从前轮优化后的best继续。仍额外10轮，batch2、Adam5e-5、7/9里程碑、dropout0.1；control三前缀均值，candidate权重0.6/0.3/0.1且率代理系数0.01；冻结编码器/ANN BEV连续头/BN，按第三前缀AP70（含epoch0）选best。

仅训练命令的随机种子改变；评估seed继续固定20260929。训练种子控制本轮数据增强、加载器顺序和消息dropout；模型初始化及上游预训练种子不变，所以只能称固定来源的微调稳定性复核。启动记录明确training_seed与evaluation_seed，配置/initial哈希必须与同组原审计一致。

复用前轮已通过的真实模型、接口、梯度审计（代码和目标不变），不声称为新seed重新做过真实梯度审计。新改的启动器只替换train命令seed，有本地/服务器定向测试：默认命令不变，zero与五模式评估seed不变，只有唯一train命令变为20260930，非法种子拒绝。每组仍重新执行全量zero与CPU复核，之后10轮训练，再p1/p2/p3/off/full完整真实包和CPU BEV/3D复算，检查冻结至第10轮与模型/GT/配置哈希。

目录`/data1/tzm/workspace/diagnostics/e3dsnn_joint_objective_<control|prefix_rate>_seed30_20261020`。从对应`e3dsnn_joint_objective_<组>_20261020_audit` prepare；调用`launch_e3dsnn_joint_round --training-seed 20260930 --allow-shared --memory-fraction .2`。181优先低利用率、显存充足GPU，按连续5次采样准入；允许两组共享同卡，阶段切换再次检查，不终止他人任务。训练每组6h、每项推理/CPU复算1h，失败无自动重试。

退出：报告两次微调种子的逐种子结果及均值/范围，不把2个种子称显著性证明。若通信下降但AP优势不复现，明确只保留节省字节的证据；若两轴改善均复现，作为当前较优低通信联合候选，后续接收电流映射优化优先从该目标出发。无论正负，完成这一次配对复核即结束该小阶段；不默认加第三种子、改系数或追逐单帧误差。下一步仍以提升联合SNN协作AP为目的，对有局部上下文的脉冲电流映射做有预算的匹配实验。

当前：服务器验证已通过，两组已启动零适配流水线，详见下方执行记录。

## 两组seed30流水线已启动（2026-10-20，最新）

代码668a0a5已push并在181 main同步，本地和服务器14项检查通过，含训练/评估seed隔离和原默认命令不变。重新prepare后的审计对象与对应首轮审计逐项相同，initial文件SHA仍逐字节等于原联合best；复用真实模型审计，没有声称重新做seed30梯度审计。

control与prefix_rate分别由coordinator28321/28558启动，共享GPU2；两次准入各5次采样利用率均0%、空闲24252MiB，每进程20% PyTorch分配上限。启动元数据明确training_seed20260930、evaluation_seed20260929，并保存真实训练命令。当前两个round_status均`running/zero`、`training_started=false`，零适配和CPU复核通过后各10轮训练，尚无新种子AP结论。

证据：[启动与训练命令](results/joint_objective_seed30_20261020/launches.json)、[初始状态](results/joint_objective_seed30_20261020/initial_status.json)、[14项检查](results/joint_objective_seed30_20261020/tests.log)、[与远端一致SHA](results/joint_objective_seed30_20261020/sha256.json)。

下一次自动任务读取两个`e3dsnn_joint_objective_<control|prefix_rate>_seed30_20261020/round_status.json`及pipeline；最终必须五模式结果完整、CPU精确复算和冻结至第10轮通过，再合并两个微调种子的逐种子表。源权重固定使这不是全训练链独立初始化实验。无关键变化静默；90分钟间隔保持，不重复prepare/launch。
