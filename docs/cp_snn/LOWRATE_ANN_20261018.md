# ANN低通信区间参照：三点有界补充（2026-10-18）

## Material Passport

来源：已复核的13点通信实验；用户授权推进SNN前端/融合精度—通信研究。此方案在查看开发验证集结果后登记，是新一轮探索，不是独立测试集确认。当前为预登记，审计/启动证据在下方追加。

## 问题、控制与退出条件

ANN2bit在22.94 kB达到60.92188% AP70，而SNN最低消息为16.01 kB。尚不能判断16 kB附近相对表现。本轮只补ANN2bit保留75%、50%、25%体素三点，覆盖更低通信区间；实际bytes完整测量，不以体素比例冒充字节预算，不插值AP冒充实测。

保持上一轮ann_full的hybrid_aa_adapt10权重、ANN融合/BEV、连续检测头、数据/GT/后处理和统一容器。选择规则与SNN相同：原始发送特征L1体素得分、坐标打破平局，保留ceil(fraction*N)；量化尺度从保留集合逐通道计算并随包发送。删去体素填零，coverage仍为原几何有效域。无参数更新、无阈值搜索、无检查点选择。

原13点全部完成且summary通过后，进行两帧真实输入的量化消息与直接重建输出精确核验，配置、检查点及源summary哈希归档。每点1738帧、BEV/3D AP、逐帧bytes和CPU精确复算/GT哈希，与原13点使用同一协议。

预算：3次全量推理及复算，0轮训练，单GPU0（先检查空闲），每子任务1小时超时、失败不自动重试。完成三点即结束，不继续搜索比例或位宽。先完整报告，再决定前端/融合联合适配或消息可用性建模；不外推未测试训练方法。ANN已有优势必须保留，不以个别SNN有利点替代整条曲线。

入口launch_e3dsnn_rate_probe的lowrate协议；独立根目录/data1/tzm/workspace/diagnostics/e3dsnn_lowrate_ann_20261018。initial协议仍13点，lowrate仅ann_full三点，新增测试验证预算与非法组合。自动任务继续读取研究总目标，90分钟、无变化静默。

## 审计及启动（2026-10-18）

本地36项回归检查通过，181的6项消息/协议检查通过；低通信三点的两帧实际序列化检测输出与直接重建特征精确一致。三点两帧消息分别为18065/18221、14240/14258、9418/9517 bytes，只是接口探针，不能外推全量AP。

原定GPU0空闲检查发现其他任务占用，审计/评估尚未启动，因此改选空闲GPU4，不停止其他任务。实际运行提交629921bb7cc69f0dcc9941d885f0c53169a9fff3，监督进程PID30330，当前q2keep75运行；三点依次评估和CPU复算。没有重试已失败实验，也未新增训练。

证据：[audit.json](results/lowrate_ann_20261018/audit.json)，SHA256 e8c2e5e410427516b5f3da6ca2cc27394546fe9c2a54dc2e05e813428b74508b；[launch.json](results/lowrate_ann_20261018/launch.json)，SHA256 334e2da513b827b552cd977a5aa517650dc387d055cc0e3bd0be17d48e5da9c0；[processes.json](results/lowrate_ann_20261018/processes.json)，SHA256 ec2bb3eebe878f41b91342a00cd413972434582bc0d4fcbe777f4b2110578db4；[startup_20261018.json](results/lowrate_ann_20261018/startup_20261018.json)，SHA256 9542929935a253b98830df41618f2ebc1c8f18aced3b99f4456a3d4279d6bb12。启动快照仍在加载，未输出帧，不将PID作为完成证据。

续接读取ann_full_status.json，若complete再取ann_full_summary.json并与原13点汇总；若running只检查健康状态。完成后停止该三点探针，不扩展无界比例搜索，按总目标选择前端/融合改进。
