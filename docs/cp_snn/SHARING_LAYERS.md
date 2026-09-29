# E-3DSNN 架构核对与共享层选择

日期：2026-09-29。论文概念、官方检测实现、当前 DAIR 移植结果分开描述。

## 架构核对

1. SVC 输出脉冲表征是论文的概念描述。实现先做体素化和连续 MeanVFE，再经过卷积、BN、Multispike。
   MeanVFE 输出不是脉冲；也不能假设名字对应 SVC 的整个代码模块出口一定已经离散化。
   本实现 conv_input 最后是 BN；表中的 early_spike 特意选在其内部第二次 Multispike 后、第三次卷积前。
2. 残差连接跨越连续膜电位/激活是正确的。BasicBlock 的分支经过 SN→卷积→BN，再与连续捷径相加。
   残差块出口因此不保证 0..4；后续遇到 Multispike 才重新离散化。代码也没有跨查询持续膜状态。
3. 论文正文写下采样 kernel=2、stride=2；但固定官方检测类 VoxelBackBone8x_3dv_snn 的 DonwBlock 实际为
   Multispike→SparseConv3d(kernel=3,stride=2,padding=1)→BN，当前移植保留它。
   基础残差块的 SubMConv3d 是 3×3×3、stride=1。末端 conv_out 则为 kernel=(3,1,1)、stride=(2,1,1)。
4. “全部推理都是稀疏 AC”过强。论文理论模型已区分第一层 MAC 和后续脉冲 AC，并采用 45nm 的
   4.6 pJ/MAC、0.9 pJ/AC 估算。当前 GPU 使用 float32 spconv/PyTorch，未实现原生事件驱动 AC 内核；
   conv_out 接收连续残差输出，2D 卷积、BN、连续检测头和解码等也不能视为纯脉冲加法。
   稀疏空间索引不等于自动跳过所有数值为零的通道，理论能耗也不是本机功耗实测。

来源：[论文方法及能耗公式](https://arxiv.org/html/2412.07360v1#S3)、
[官方固定提交的检测主干](https://github.com/bollossom/E-3DSNN/blob/dbe5d1731d3204850bd591a18f7255ca2d1b6712/det/pcdet/models/backbones_3d/spconv_backbone_spike.py)。

## 当前配置下的候选大小

本次新增实测：SNN 第 19 轮固定权重、相同 64 个等距验证帧、batch=1；不是原论文 KITTI 的张量尺寸。
输入体素 [0.2,0.2,0.1] 米，范围 [-100.8,-40,-3.5,100.8,40,1.5]。
下表 MB/KB 为十进制；M 随帧变化，列出平均值取整。FP32 仅特征数据，不包含稀疏坐标。
打包列是实际 bytes，包含本次 codec 的地址、掩码、计数、JSON 元数据和 CRC。

| 位置 | 特征形状（不含 batch） | 值类型 | FP32 特征平均 MB | 平均计数包 KB |
| --- | --- | --- | ---: | ---: |
| 早期 SVC 内部第二次 Multispike 后 | [M≈27,496,16] | 0..4 计数 | 1.760 | 202.50 |
| x_conv2 | [M≈37,424,32] | 连续残差/膜电位 | 4.790 | 不适用 |
| x_conv3 | [M≈21,820,64] | 连续残差/膜电位 | 5.586 | 不适用 |
| x_conv4 | [M≈9,008,64] | 连续残差/膜电位 | 2.306 | 不适用 |
| conv_out 最终 Multispike 后 | [M≈6,846,128] | 0..4 计数 | 3.505 | **155.30** |
| HeightCompression 后 | [384,50,126] | 0..4 计数 | 9.677 | 188.47 |
| BEV 主干最终输出 | [256,50,126] | 连续值 | 6.451 | 不适用 |

前三个残差尺度的三维网格 z/y/x 分别为 [26,200,504]、[13,100,252]、[7,50,126]；
conv_out 为 [3,50,126]。其稠密等价形状为 [128,3,50,126]，完整 FP32 网格为 9.677 MB，
与实际稀疏特征数组的平均 3.505 MB 不同。

“不适用”指不能直接无损使用此 Count4 codec；并非不能共享或不能量化。
若要共享中间脉冲，应定位已有神经元之后的确切张量，或把额外量化作为独立实验。

HeightCompression 在当前代码仅 dense+reshape，将三个高度格折入通道：
128×3×50×126 → 384×50×126，没有高度池化，也没有减少特征值数量。
但重新按 BEV 位置组织后，稀疏地址/通道掩码开销改变，因此同一 codec 下包反而比 3D 地址组织略大。
这里统一沿用 uint16 z/y/x 地址，包括 BEV 中恒为零的 z；不能把该结果当成最优 BEV codec 的压缩下界。

## 建议与 155 KB 的准确含义

首选 conv_out 最后一个 Multispike 之后、HeightCompression 之前的 encoded_spconv_tensor。
它保留了高层语义、3D 地址和原生离散计数，且接收端还可以继续运行 BEV 主干和检测头。
不必为了得到可打包值域另加一个通信编码器。建议链路为：
单端提取 → 筛选/打包 → 解包 → 转换到 ego 3D 坐标 → 融合 → HeightCompression → BEV 主干 → 检测头。
几何变换/插值可能产生连续值，需明确作为融合输入电流还是经过重新发放；不能自动声称全链路低比特。

155.30 KB 是**每个发送端、每帧整个末端特征**的平均包，不是单个体素、单个通道、最终检测框或完整网络流量。
每帧约 6,846 个稀疏活跃位置；剔除全零向量后约 6,036 个位置，每个位置保留 128 维计数。
编码包含 6 字节 z/y/x 地址、128 bit 通道掩码及每个非零值 3 bit，外加实际包头/元数据/CRC。
这个候选就是 [PACKET_AUDIT.md](PACKET_AUDIT.md) 中无损回放验证的共享点。

BEV 折叠后是有价值的第二候选，方便使用已有 BEV 融合模块，但需要区分 height-folding 与真正丢弃高度，
以及平面 SE(2) warp 与完整 SE(3) 对齐。早期脉冲层可作为高分辨率对照，语义较浅且接收端计算更多。
最终连续 BEV 特征可作为标准 ANN 风格通信对照；目前没有证据支持把它直接叫原生脉冲消息。

当前单阶段适配可让共享末端进入完整后续检测链路；若恢复原论文 Voxel R-CNN RoI 精修，
其多尺度 x_conv2/3/4 是否获得协作者信息必须另行设计，不能假定共享末端就覆盖原版两阶段检测器。

## 证据

脚本 `opencood/tools/audit_e3dsnn_share_layers.py`，测量提交 aff5858。
服务器目录 `/data0/chen/gzc/workspace/diagnostics/e3dsnn_share_layers_epoch19_20260929`。
仓库 `results/share_layers_{summary,manifest,status}_20260929.json` 和 `share_layers_frames_20260929.jsonl`。
这些是当前配置和固定阶段权重的测量值，不是训练后期或协作系统的最终结果。
