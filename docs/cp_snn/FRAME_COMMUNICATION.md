# 逐帧通信量审计（2026-10-11）

网络恢复后，在服务器同步 main，复用首轮融合已冻结的 SNN/ANN best@27，单独补跑完整验证集。
没有重新训练，也没有覆盖 2026-09-30 的最终评估。本次推理代码为 `1898f21`，包含逐帧记录改动 `4b0da2b`；
导出工具为 `opencood.tools.export_e3dsnn_communication`（`96f219e`）。

**当前状态：后台补算已启动，结果尚未取回核验。** 初次 SSH 登录和代码同步成功；
最后观察到 SNN 全量推理结束、CPU 复算运行中，ANN 推理超过 1,250 帧。
随后 SSH 会话被重置，两次重连均在 banner exchange 超时。
补算由独立监督进程运行；不能据 SSH 断开判断任务失败，也不能未经检查宣称最终完成。

## 字节口径与明细格式

每帧包含一个路端到车端的已编码特征消息。数值是实际序列化对象的字节长度，
包含稀疏坐标、特征、格式头、时间戳和变换等元数据；不包含网络分片、链路/传输层头部、重传和确认包。
这不是原始 FP32 稠密张量大小，也不是整网实测流量。
SNN 使用无损 Count4 编码，ANN 使用无损 FP32 压缩 NPZ；不是等码率量化对照。

导出完成后的 `frame_communication_20261011.json` 将在 `per_frame` 数组逐行记录：

- `index`：验证序号，从 0 开始。
- `frame_id`：原始字符串帧号，保留前导零。
- `snn_bytes`、`ann_bytes`：该帧各模型实际发送的总字节数。

报告的 `models` 保存每种模型的完整统计、评估结果、CPU 复算结果和推理 manifest，
包括权重 SHA256、数据清单哈希和预测索引 SHA256，可追溯到同一份逐帧记录。
单车不发送协同感知特征，特征通信量仍为 0；该零值不代表整个车辆网络没有其他流量。

## 复核规则与复现位置

导出前要求推理和 CPU 复算完成，AP/3D AP、样本及框数一致，预测索引哈希有效。
两组必须有 1,738 个不重复且顺序相同的 frame_id，数据清单与划分哈希一致，
每帧恰有一个正整数字节数的路端包。逐帧求和、平均、中位、最小、最大和 P95 必须与 evaluation.json 精确一致。
本次使用 GPU 6/7；同服务器其他 GPU 有任务，因此不将运行耗时当作独占性能基准。

服务器目录均在 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_fusion_20260930_v1`：

- `count4_framebytes_20261011/`、`relu_framebytes_20261011/`：新推理快照，含完整预测和逐帧索引。
- `*_framebytes_20261011_pipeline.json`：评估、复算的完成/失败状态。
- `*_framebytes_20261011_plan.json`、`framebytes_20261011_launch.json`：命令、GPU、权重哈希和进程来源。
- `frame_communication_20261011.json`：经验证导出的配对逐帧明细。

监督 PID：SNN 2099564、ANN 2099567；运行计划包含实际子进程命令和 1 小时/阶段超时。
独立的 SSH 监控/导出会话已因连接重置中断，导出文件是否已经生成尚未确认。
恢复后先检查这两条 pipeline 和报告是否已存在，禁止重复启动同一补算；
若两条 pipeline 均 complete 且报告不存在，再运行下述导出命令。
若已有报告，核验后取回；随后同步 docs/cp_snn/results 与用户 evidence 目录。
当前定时任务保持此前 PAUSED 状态，本次未修改用户的暂停设置。

导出示例（使用上述根目录下的绝对路径）：

```bash
python -m opencood.tools.export_e3dsnn_communication \
  --snn-dir <count4_framebytes_20261011> \
  --ann-dir <relu_framebytes_20261011> \
  --output <新的报告路径> --expected-frames 1738
```
