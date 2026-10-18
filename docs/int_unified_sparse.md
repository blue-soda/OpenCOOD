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
