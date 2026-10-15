# GRU 参数与缓存匹配的混合 SNN 原型

## Material Passport

- 任务：2026-10-15 用户要求构造与当前 GRU 架构、参数量匹配的 SNN 记忆模块。
- 范围：实现 `spike_gru`，验证语义及真实数据两轮工程试跑；本文件不是精度改善报告。
- 数据：既有 DAIR 自车完整片段；与第一轮相同的冻结清单及 P1 初始化，无异步融合。
- 状态：实现已准备，执行验证后补充结果。

## 匹配范围

严格复用原 INT GRU 的层定义与参数名称：reset/update/candidate 均为 128→64 的 3×3 Conv+BN，前两者使用 Sigmoid，candidate 使用 Tanh；读出为 66→64 的 3×3 Conv+BN+ReLU。64 通道膜电位取代原隐藏特征，另保留原有 age 通道，无额外 trace、可学习阈值或时间常数。

记忆模块共 259,712 个参数，整个模型共 30,776,212 个参数。200×504 BEV 上 FP32 缓存为 65×200×504×4 = 26,208,000 字节，均与当前 GRU 完全相同。继承同一组空间网络、同一位姿 warp/重置逻辑、同一检测损失与训练器；相同 RNG seed 下各层初始化也相同。

匹配的是有参数的层、通道、初始化和缓存预算，**不是完全相同的计算图或 FLOPs**。门控、候选卷积及检测器保持 ANN；新增的是发放、复位和代理梯度。这是混合 SNN，并非全脉冲 GRU 或事件驱动 GPU 稀疏算子。

## 状态更新

输入 X，位姿对齐后的上一帧膜电位 U，门控与候选仍沿用原 GRU：

```
r = sigmoid(BN(Conv_r([X, U])))
z = sigmoid(BN(Conv_z([X, U])))
c = tanh(BN(Conv_c([X, r * U])))
V = z * U + (1 - z) * c
S = H(V / theta - 1) - H(-V / theta - 1)
U_next = V - theta * S
Y = original_readout([theta * S, age, occupancy_mask])
```

S∈{-1,0,+1} 是正、负两种二值脉冲的紧凑表示，每个真实扫描一次更新。保留双极性是因为原候选使用 Tanh，避免直接丢弃负值信息。theta=0.25 是固定超参数，在专用 YAML 中记录，不增加参数；由于候选有界，不使用会导致难以发放的默认阈值 1。选择 0.25 是工程起点，未按验证 AP 搜索。

读出使用脉冲，不能直接读取 U 或 V。原 GRU 已有的 age 与 occupancy_mask 两个辅助通道保持不变，因此也不能声称检测输出完全由脉冲决定。复位可微，二值阶跃使用既有 atan 代理梯度。连续 U 可以被门控读取，缓存仍为 FP32，不宣称存储压缩或节能。

## 验证与运行

语义测试 `opencood/test/test_int_spiking_gru.py` 覆盖：参数/层/初始化逐项匹配，双极脉冲，亚阈值积累与复位，无膜电位读出，跨帧代理梯度及 detach 边界，位姿对齐/场景重置/状态不变性和阈值合法性。同时重跑原 INT 上游等价性和 LIF 测试，防止基础状态接口回归。

配置：`opencood/hypes_yaml/dair-v2x/snn/int_ego_spiking_gru.yaml`；训练入口 `run_int_epochs --mode spike_gru --tbptt-steps 4 --fm-only-epochs 1`。原配置和历史实验不改动。真实数据试跑与新 GRU 对照使用同一新 YAML、同一 P1 初始化、seed303、相同两段 train/val，跑两轮以覆盖 FM 预热和联合训练；试跑 AP 不进入科学比较。尚不启动全量训练。
