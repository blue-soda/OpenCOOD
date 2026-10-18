# 统一BEV状态更新：变化区域重算与不变区域复用

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan / run
- Origin Date: 2026-10-19（Asia/Shanghai）
- Verification Status: 71项INT测试通过 / 调度短测完成 / 真实序列试验已提交
- Version Label: unified_selective_update_v1

## 目标和范围
用户授权优先验证状态更新模块的局部执行，关注真实精度与延迟并使用空闲GPU。v2空间编码/上下文和读出仍每帧完整计算，先针对占核心Conv2d MAC约65%的19.00G状态更新模块。现有v1续训、v2时序/v2单帧训练保持运行；不修改运行中的模型或训练契约。

本轮是冻结权重的推理干预，不声称局部训练已完成。v2目前很早期，先用固定epoch1验证实现和耗时，再对训练更充分的权重重复全5794扫描/1738标签评估。早期前缀AP既不是完整AP，也不能证明架构精度达到工程门槛。已有75.82349%三级缓存候选继续保留。

## 因果选择与缓存
- 当前证据为v2空间编码128通道+上下文128通道，当前证据完整计算后再选择区域；选择器不调用完整状态更新、不读GT/教师预测。
- 缓存单份状态256通道、对应位置**最近一次状态更新时**的证据256通道及一张age图。三张图始终在上一处理帧坐标下，拼接后用同一SE(2)最近邻warp对齐；状态/证据未更新的位置仍随自车运动对齐。
- 每个tile的分数是当前与缓存证据的RMS差，除以两者较大的RMS幅度。证据仅在实际状态更新的位置替换，避免每帧小变化被相邻帧比较漏掉。
- 分数超过训练集阈值则请求更新；新视野、age>=0.5s强制更新。沿用场景/显式/间隔>0.25s重置和非因果拒绝。
- 对选中tile合并矩形，带两格halo裁剪，按相同裁剪形状分批调用更新模块。计算输入面积（包含重复halo）>=整图0.8时回退整图。未更新区域保留对齐状态，最后整图执行逐点读出/检测头。
- 缓存从25,804,800字节增至51,710,400字节，额外warp、分数计算、年龄维护和裁剪调度均计入模型耗时。还没有目标运动预测、稀疏前端、上下文低频缓存或选择器微调。

## 对照设计与训练集校准
同一不可变v2时序checkpoint，每个GPU顺序运行：原整图before → 实际局部q50 → 整图算后局部写回q75 → 实际局部q75 → 原整图after。最后一个整图用于检查同GPU耗时漂移，AP应与before一致。

q50/q75分别为训练集固定前1024扫描、覆盖有效tile分数的50%/75%分位数。校准时正常每帧更新，记录帧顺序与源权重SHA；不读取验证AP调阈值。进入局部递归后参考特征会积累年龄，故分位数不是固定更新预算。首轮分块16/32，max_age=0.5s、fallback=0.8事先固定，不根据验证AP偷偷更改。

整图算后局部写回与实际局部执行使用相同区域规划及fallback规则，差异只在执行方式：前者完整计算再where写回，后者仅调用裁剪区域。相同输入裁剪与整图输出等价用测试验证；真实序列核对两种方式的写回比例与fallback逐帧一致，分别记录AP差异，不默认小数值差异一定不会影响NMS。

| 阶段 | 计划GPU/输出目录 | 目的 |
|---|---|---|
| tile16早期试验 | 181 GPU5 / unified_sparse_e1_t16 | 512验证扫描的实现与延迟诊断 |
| tile32早期试验 | 181 GPU6 / unified_sparse_e1_t32 | 较大块降低halo/调用数是否值得 |
| 后续完整评估 | 空闲GPU，固定较充分训练的权重 | 完整AP30/50/70、均值/P50/P95、复用率、实际裁剪面积、回退率和缓存量 |

GPU5/6各自的dense与sparse顺序配对；不能直接比较GPU5和GPU6的绝对耗时来挑赢家。其他GPU仍训练，主机争用可能影响计时，本轮为筛查，不代替稳定条件下最终性能确认。若局部不加速，即使理论MAC减少也不宣布有效；根据碎片化/调用数优化批处理或调整执行方式。若策略本身掉点明显而裁剪等价成立，才考虑选择策略和选择性更新训练，不能把它误诊为裁剪实现问题。

## 工程与执行
新增SelectiveUnifiedBEV和run_int_unified_sparse。测试必须覆盖：全更新包装器与原模型预测一致；未选区域不调用更新模块且保留参考特征；累计age强制更新；新视野；非因果/场景重置；实际裁剪和整图后局部写回的状态/预测一致。沿用refresh_tiles并增加masked_dense执行选项，原调用默认行为保持。

本地main提交/push，181 pull、测试后启动。源checkpoint需对应epoch报告存在并核对SHA才加载，最多等待30分钟，失败保留现场。本轮固定epoch1，不能在运行时自动追逐best改变权重。

```text
python -m opencood.tools.run_int_unified_sparse --root /data1/tzm/workspace/diagnostics/int_spiking_gru_20261016 --checkpoint /data1/tzm/workspace/diagnostics/int_cache_repair_20261017/unified_bev_v2_recurrent/epoch_001/student.pth --output /data1/tzm/workspace/diagnostics/int_cache_repair_20261017/unified_sparse_e1_t16 --gpu 5 --tile 16 --frames 512 --calibration-frames 1024
python -m opencood.tools.run_int_unified_sparse --root /data1/tzm/workspace/diagnostics/int_spiking_gru_20261016 --checkpoint /data1/tzm/workspace/diagnostics/int_cache_repair_20261017/unified_bev_v2_recurrent/epoch_001/student.pth --output /data1/tzm/workspace/diagnostics/int_cache_repair_20261017/unified_sparse_e1_t32 --gpu 6 --tile 32 --frames 512 --calibration-frames 1024
```

输出包含launch/provenance/engineering/selective_engineering/calibration/experiment_contract、逐帧记录、各对照report和总report。读状态和日志核验实际执行，不以进程启动等同实验完成。现有路线级定时覆盖所有训练与局部试验；在短实验完成节点适当缩短间隔，稳定长训练时再放宽。

## 2026-10-19 调度短测已发现碎片化瓶颈
实现a1e40af、微基准2b255f7及固定块改进43f9c41均经本地main提交/push、181 pull；原70项及新增固定块批处理等价共71项INT测试通过。真实序列GPU5 PID31448（tile16）、GPU6 PID31625（tile32）已提交，等待v2时序epoch1不可变checkpoint及完整epoch报告写完后自动执行，最长等待30分钟，不修改源训练。

同时在空闲GPU7完成两个短测unified_update_dispatch_v1和unified_update_dispatch_fixed_v1，输入为随机513×100×252 packed特征、随机初始化的实际ResidualStateUpdate，seed303；每种条件预热5次，同步计时20次。仅测更新模块与裁剪规划/调度，不含观测、选择器、warp及读出，无检测精度含义。

| 25%请求tile的场景 | 整图更新约ms | 原合并矩形裁剪ms | 固定块batch32裁剪ms |
|---|---:|---:|---:|
| tile16连续成片 | 6.65–6.71 | 5.195 | 10.721 |
| tile16零散 | 6.62–6.74 | 18.660 | 11.518 |
| tile32连续成片 | 6.72–6.76 | 3.099 | 8.488 |
| tile32零散 | 6.66–6.67 | 11.432 | 10.462 |

原tile16零散裁剪实际输入面积仅整图36.9%，仍明显更慢；面积不足以预测延迟。固定块降低部分碎片化调度开销，但重复halo和多组边界形状增加代价，依然未胜整图且损害连续区域表现。固定块实现作为显式layout=fixed、crop_batch=32可选对照保留，原默认merged/batch8保持；不因“有优化代码”就宣称加速，也不据此新增固定块全量训练。GPU7短测完成后释放。真正序列仍先执行原tile16/32试验，后续按实际区域布局/耗时判断是否需限制碎片数、按延迟回退或换执行内核。

两个report及launch归档docs/int_spike_results/cache_repair_20261017/unified_update_dispatch{_fixed}_v1_*。路线级定时暂从60分钟改20分钟以捕获短试验完成，提示词不变“推进时序感知，仅完成异常通知”；短试验结束后回到60分钟或根据5/10轮训练节点调整。

同轮真实启动确认：GPU5/6两条试验均已完成等待并通过真实权重工程验收，源epoch1 SHA=46d336a1dba40f0fd1824d01400f8edd5c4962f84d4d7b261de7de652b14db7f；各8帧全更新包装器预测误差0、实际裁剪选中区误差0。训练前1024帧校准已完成：tile16 q50/q75=0.15568511/0.21653467，tile32=0.16081000/0.21256878，已进入dense_before真实序列评估。归档unified_sparse_launch_audit_20261019.json及两条engineering/selective_engineering。源v2 epoch1完整AP70仅3.36844%，明显仍处于早期学习；本轮前缀结果仅用于执行/机制筛查，不可当作可用模型的精度保持证据。后续在v2第5/10轮按既定节点同时检查时序/单帧学习曲线，再决定较充分权重上的完整局部评估，不在本轮据单个早期epoch下架构结论。

### 2026-10-19 04:17 局部执行首轮完成：正确但未加速
unified_sparse_e1_t16/t32均COMPLETED。相同epoch1 SHA=46d336a1dba40f0fd1824d01400f8edd5c4962f84d4d7b261de7de652b14db7f，训练1024帧校准、验证前512扫描/255标签；逐帧顺序、权重、年龄与写回比例/fallback、两次dense指标一致性核验通过。masked_q75与sparse_q75的AP30/50/70均精确一致；证明该前缀没有可见的裁剪数值影响，不能推断其他权重/全序列都等价。归档两条report及unified_sparse_completion_audit_20261019.json；不是从保存框独立重算AP。

| 分块/模式 | 前缀AP70% | 平均ms | P95 ms | 实际写回比例 | 裁剪实际输入面积比例 |
|---|---:|---:|---:|---:|---:|
| tile16 dense before / after | 5.1979 / 5.1979 | 21.654 / 20.787 | 25.371 / 24.135 | 100% | 100% |
| tile16 q50 sparse | 4.8456 | 36.878 | 51.773 | 69.10% | 79.68% |
| tile16 q75 masked | 4.4286 | 23.646 | 27.834 | 48.66% | 100%（完整计算） |
| tile16 q75 sparse | 4.4286 | 35.321 | 46.418 | 48.66% | 58.07% |
| tile32 dense before / after | 5.1979 / 5.1979 | 21.222 / 20.589 | 25.142 / 24.038 | 100% | 100% |
| tile32 q50 sparse | 4.7004 | 27.353 | 35.251 | 72.91% | 78.14% |
| tile32 q75 masked | 3.3547 | 23.269 | 26.514 | 54.27% | 100%（完整计算） |
| tile32 q75 sparse | 3.3547 | 27.697 | 34.806 | 54.27% | 59.73% |

整图前后耗时漂移分别-4.00%/-2.98%，明显小于局部版约30%–70%的变慢。两个GPU各自顺序配对，其他GPU仍训练，且为早期低精度权重；该表仅512前缀，不能与完整75.82%候选混排，不宣称正式生产速度或最终精度损失。当前策略可复用约46%–51%的状态，但实际延迟未获益；不扩大这版碎片化裁剪的全量实验。

为定位成本，在GPU5启动并已完成unified_sparse_stage_profile_e1（PID47310，源码76d9f02），固定同一权重/训练阈值，256扫描、去前10帧。新增profile_int_unified_sparse_stages.py只加计时，不改权重/策略。CUDA事件跨度诊断：dense总墙时20.48–21.60ms，当前证据9.88–10.58ms（其中上下文7.08–7.64ms、观测2.59–2.70ms），更新6.17–6.24ms，Pillar2.13–2.34ms、warp0.97–1.04ms、读出约1.04–1.08ms。故65%的卷积MAC只对应约30%的当前模型墙时，不应再按MAC比例估计实测加速上限。
局部tile16更新调用平均9.63次/帧、累计更新事件18.93ms、整个裁剪执行20.42ms；tile32为4.78次/帧、10.87ms、11.78ms。masked32仍一次更新约6.30ms，选择分数约0.51ms，整步23.16ms。这支持“小卷积调用碎片化”是主要额外负担，同时约2–3ms的选择/缓存管理成本也需承担。事件跨度可能含CPU调度导致的GPU空闲，嵌套项不能相加，不是纯核函数时间。报告归档unified_sparse_stage_profile_e1_report.json，短测GPU5/6/7已释放，不再轮询已完成试验。

下一步执行侧优先验证按选中位置批量收集、固定次数矩阵运算的专用更新执行器，避免把一个大卷积拆成多种形状小卷积；先同输入边界/裁剪等价和微基准，若仍不比整图快则停止这一实现方向，再考虑更粗的连通区域或重设更新算子。不能仅加大分块、降算量或调验证阈值后宣称成功。精度侧继续在v2第5/10轮比较时序/单帧/v1相同轮次曲线，成熟权重上的完整稀疏AP验证以执行器先有实际速度空间为前提。

训练增量审计：v1续训41–45、v2时序/单帧1–3均完整8593训练扫描/4678标签/2262窗口、5794验证扫描/1738标签，顺序及权重一致，归档unified_parallel_progress_20261019_0417.json。v1最佳epoch45 AP70=58.35247%、LR2.5e-4/stale0，仍远低于可用目标；v2 epoch3时序8.96422%、单帧13.43730%，均在epoch4，早期改善中，尚不能证明v2优于v1或时序有害。
审计额外发现v2单帧epoch3仅一个窗口（终帧005904，所含005901–005904）loss=320012.67、裁剪前梯度范数336691.72，拉高epoch均值至142.45；其余2261窗口均值0.98077，前后窗口正常，原日志及均值完整保留。既有代码梯度裁剪上限5、无非有限值，该现象不能忽略也不等于持续发散。GPU6用epoch2/3末权重重载这4帧，均有11–29个正anchor、GT目标幅度正常、GT loss约1.33–1.69，未复现尖峰；诊断不拥有原窗口更新前权重/完整随机状态，不能据此宣称异常已解决或归因坏标签。证据unified_v2_reset_e3_loss_probe.json。继续观察是否复现；后续新训练/续训可加异常窗口快照与损失分项，不能擅自剔除该样本。
当前路线唯一活动训练仍GPU0 v1、GPU3 v2时序、GPU4 v2单帧。读取到现有automation间隔40分钟，本轮保留以覆盖第5轮检查节点，提示词仍“推进时序感知，仅完成异常通知”，不新增任务。
