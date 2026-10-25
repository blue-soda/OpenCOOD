# InfoCom低预算适配准备：真实梯度与数据范围检查

## 状态与总目标

2026-10-26，PREPARATION / real-batch audit passed，非正式训练、非AP实验。目标仍是方案二SNN协作对强ANN的通信—AP—计算折中，方案三暂停。冻结InfoCom曲线已结束，不能通过未适配低比例退化来宣布SNN独有优势；本项准备给强ANN匹配适配机会。

## 已实现并运行的检查

代码`opencood/tools/strong_ann_baselines/audit_infocom_lowrate_training.py`，实现87b48dd、数据清单修正fbc58f0。本地语法检查通过，181 GPU4五次准入均24252MiB空闲/0%利用率后执行，600秒上限，无优化器步、不输出大checkpoint、不影响现有多预算训练。

固定官方epoch7权重SHA 7375da04507b4f543666e15aa02897e41e3c32e08bcb3f74b27ad49fb28c803c，原sigma→std兼容映射仍严格限定；新增AdaptiveScale保持identity且冻结，避免把新参数变化误作官方结构。model处于train，保留mu+eps*std重参数随机latent、原mask评分与4bit STE；只显式固定top-k比例为1%或10%。全BN running buffers冻结且结束逐值核验未变化，这属于明确声明的适配设置，不称未经修改的官方训练。

实际增强训练样本000010/000011，batch2，每帧两参与方；mask输入[4,1,200,504]，1%选4032个非零位置，10%选40320个位置。共10组梯度（发送encoder、decoder、mu/std、mask两层、BEV backbone、cls/reg/dir heads）全部有限且非零。1%检测损失1.4416325，10%为0.45370045；这些只是两次不同随机latent的梯度检查，不能把单批损失差当成最终精度比较。KL仅记录562.76538，未加入本次审计loss。峰值CUDA allocated 4,638,149,120 bytes，约4.64GB/4.32GiB；不含Adam状态，不能直接视为正式训练总显存预算。

## 数据保护发现与修正

第一次审计在任何模型训练前停止：错误地把原始train.json的4811个ID预期为4678。原SNN的paired loader会根据pair_manifest.valid_ids筛成4678，因此既有SNN训练数据未受影响；这是新ANN审计入口的假设错误。保留gradient01失败，不覆盖为成功。

第二次审计显式校验原始split SHA、paired manifest协议与cooperative/data_info SHA，从同一manifest导出4678个有序有效ID，生成common_train.json并核对官方dataset实际加载顺序。新有效split SHA dae42ad7dfbc6e917c6630279fd2f54f64efac8b62825f3686dc75e4a5da4788；源4811-ID SHA 805f09b1bd0b80cde81aa3e01983afa21f7796712db7c534ed0335c7842438aa。此后正式适配必须复用此清单，不能直接把官方4811帧当成匹配数据。

## 尚未完成、下一步顺序

1. 将同一检查过的加载/比例覆盖接入有限轮数的适配训练器，明确训练范围、随机latent、KL启用时刻、BN、选择准则；写入正式协议后再启动。当前仅梯度审计，没有完成训练器或启动ANN训练。
2. 统一记录4678训练帧、相同适配epoch/步数/种子机会；保留官方随机latent与STE，不暗中把通信模块eval。1%/2.5%/5%/10%评估都用真实字节及共同GT；空间比例不等于2KiB硬预算。输出改变的实际通信量仍须重新测量。
3. 先解决检查点容量：181 /data1约142MB、/data2约2GB、根盘约2.1GB空闲。当前自有恢复目录约1.1GB，空间下降来自同一共享分区的总使用变化，未归因于其他具体任务。完整119.85M参数模型FP32约0.48GB，若全参数Adam加权重的恢复状态约1.44GB；last原子写入临时副本再加best就可能超过当前空间。不能以只存best、丢优化器/RNG来伪装可靠恢复。没有删除他人数据、没有启动可能写满磁盘的ANN长训练。
4. 现有SNN状态约几十MB，继续按原10轮训练并监测剩余空间；最终真实wire/AP复核之后，再对接统一计算与选择机制消融。不追加冻结ANN比例扫描或恢复方案三。

证据：[成功审计](results/strong_ann_baselines_20261025/infocom_gradient_20261026/infocom_lowrate_gradient02/audit.json)、[首次数据假设失败](results/strong_ann_baselines_20261025/infocom_gradient_20261026/infocom_lowrate_gradient01/failure.json)、[成功启动](results/strong_ann_baselines_20261025/infocom_gradient_20261026/infocom_lowrate_gradient02.launch.json)、[下载SHA](results/strong_ann_baselines_20261025/infocom_gradient_20261026/download_sha256.json)。


更新：训练器与两步Adam检查已完成，正式训练仍受存储准入阻塞；以上“训练器尚未完成”为历史状态。以[正式适配协议和准备结果](INFOCOM_LOW_RATE_ADAPTATION_20261026.md)为准。
