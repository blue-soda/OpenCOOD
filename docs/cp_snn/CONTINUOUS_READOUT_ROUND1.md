# 接收端连续残差消融（2026-10-17）

## Material Passport

来源：已核验辅助监督实验、用户授权AP70顺序；academic-research-suite / experiment-agent执行核验。
当前状态：真实batch检查通过，181 GPU5已启动30轮训练；监督PID2458，源提交867ae9b5a6a88a337b2d769fd320d6627b85e4ac。

## 唯一改动与假设

SNN辅助监督57.87040%低于预定ANN单车57.98007%。检查融合后再次Count4取整/截断是否限制连续对齐残差利用；这是待验证假设，现有描述性探针不能证明因果。
对照count4_aux1_joint30：`fused = Q(ego + alpha * supported * delta)`。
新实验count4_contaux1_joint30：`fused = ego + alpha * supported * delta`；无覆盖位置仍精确保留ego。
只取消融合输出的取整和[0,4]截断，允许带符号连续值。融合内部Count4、3D/BEV SNN主干、检测头及消息编解码不变。
这是连续接收端读出的混合SNN，不能宣称全链路纯脉冲AC。消息仍来自融合前路端3D Count4特征。
同权重切换读出时包长应相同；重新训练可能改变发送端稀疏度，实际包长需重测。

## 匹配与判定

原始SNN best@53初始化，融合零初始化；λ=1、固定BN、3轮融合预热+27轮联合，总30轮；seed20260929、batch2、Adam0.0002、20/27衰减。
对照相同GPU5及数据划分/增强/阈值。启动器强制新旧配置除name/readout外完全一致。
主对照57.87040%/91.24722 kB，过线参照仍为预定ANN单车57.98007%；另观察ANN融合权重关闭路端58.19843%，不能冒充独立单车基线。
按AP70选权重，完成后自动开/关路端评估、逐帧消息、3D AP、CPU复算和逐目标诊断。
单种子仅用于工程筛选。过线后再补严格匹配单车/多种子及量化ANN等码率对照；不以单次越线宣布研究目标证实。

## 执行

根目录 `/data1/tzm/workspace/diagnostics/e3dsnn_continuous_round1_20261017`，工具 `opencood.tools.launch_e3dsnn_continuous`。
已有启动记录拒绝重复，GPU须空闲、磁盘至少8GiB；训练12h、评估各1h硬超时，失败不自动重试。
本地21项CPU回归通过，覆盖小残差/负值/超4值、梯度、无覆盖恒等、旧ReLU行为及AP复算。


## 启动验证

21项CPU回归在本地与181通过。真实batch2预热/联合两步检查：融合及应解冻检测器具有有限非零梯度，BN统计保持不变，推理不执行辅助分支。
固定同一发送端编码结果切换两种读出，消息完全一致（两帧61136/70667字节），BEV输入确实不同。
首次检查错误地要求两步更新后最终检测输出立即不同，该断言失败；日志smoke.log保留。改为在实际改动边界检查BEV输入并记录检测输出，而非要求下游量化后立即变化。
修正后检查通过，最终检测输出在这两步仍相同（distinct_detector_output=false）；不能把小batch检查当成精度改善或完整训练的证据。
修正检查日志smoke_boundary.log与smoke.json保留，训练仅启动一次。
启动器已确认四组对照完整结束、共同GT一致、试验配置匹配、相同原始单车SHA、GPU5空闲和磁盘余量；自动任务继续30分钟跟进。
完成后调用 `summarize_e3dsnn_fusion_probe --suffix _contaux1_joint30 --audit-name fixed_bn.json --activations count4` 导出核验摘要，并与辅助监督组比较。

已观察第1轮超过400批，loss与梯度有限。启动证据 [startup_20261017.json](results/continuous_round1_20261017/startup_20261017.json)，SHA256 `072ccdcf689fec8a422ad25fa2aaa6329ad5a9a89f47fa69ebd66ea3aece2e85`。
