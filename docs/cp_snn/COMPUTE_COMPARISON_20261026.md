# 方案二与强ANN：统一计算测量协议（预登记）

2026-10-26登记。此项服务于总目标中的AP—真实通信—计算折中，不增加训练臂或超参数搜索，不恢复方案三。目前只完成边界审查及协议，尚未完成共同计时器或受控实测，不能据本文件声称计算优势。

## 当前计时为什么不能直接放在同一主表

- 方案二`train_e3dsnn_single.validate`默认排除最初20帧；InfoCom与CoSDH评估器在首帧重复3次热身，边界不同。
- InfoCom有纯forward及forward+后处理两个计时；CoSDH是完整双向hybrid调用及后处理，且真实wire审计仍包含重复decode/warp。方案二还可能在forward中导出真实包到磁盘。直接比较这些值会混入不同工作量。
- 现有运行在不同GPU、不同时间及共享服务器负载下，只有运行记录意义。方案二逐事件重打包寻找第一个超预算前缀的成本是当前实现真实成本，不能从总耗时中隐藏；同样不能利用CoSDH仅用于审计的重复计算宣称胜出。

## 固定工作点和输入

优先完成现有10轮及最终复核后冻结模型SHA：方案二固定2KiB最佳、InfoCom适配最佳1%、InfoCom官方10%、CoSDH官方完整协作。已有同结构量化ANN 2KiB作为内部参照，先核实检查点及实际执行路径再纳入，不能以未核验模型补表。多预算方案二完成后按原2KiB选模规则加入，不选择速度最快的epoch。

共同1738帧验证ID按原顺序，用`linspace(0,1737,64,dtype=int64)`固定64帧做工程测量。全部方法使用相同帧ID与有效两参与方，保留各自预处理、体素分辨率及后处理；这比较完整方法，不是相同计算图。每帧实际包及输出必须与既有完整验证参考一致。64帧仅用于成本测量，主AP仍用1738帧，不另做64帧AP选模。

同一张RTX3090、batch1、FP32与现有推理数值设置不变，固定CPU线程数1；按方法轮换顺序重复3遍，保存顺序和种子，不选择最好的一遍。每一遍20次热身调用，测64帧；热身不混入样本。先以4个固定ID审计计时边界/输出/真实bytes，不能先从随机几个高速样本得出结论。记录GPU UUID、驱动、Torch/CUDA、CPU型号、线程、显存上限、其他GPU进程与CPU负载。发现同卡其他计算任务时本次标为有干扰，原数据保留，不能悄悄删除慢样本或重试直到更快。

## 共同边界

总延迟从已准备好的CPU输入开始，依次包含H2D、两参与方编码、消息选择、实际CPU编码、CPU/GPU转换、解码、对齐/融合、BEV和头、检测解码/NMS；结束于CPU预测框/分数可用。开始/结束CUDA同步，使用wall clock。真实网络传输、数据读取/体素预处理、写文件、AP计算单独列出，不包含在该总延迟内。另报H2D后到CPU框的计算延迟，避免输入点数差异被掩盖。

分阶段记录编码器、消息生成/真实codec、接收解码/对齐/融合、BEV/连续头、后处理，注明边界是否跨CPU/GPU；分阶段同步会扰动调度，阶段计时使用单独instrumented pass，不把阶段之和冒充无钩子的总延迟。保留逐帧真实bytes、峰值allocated/reserved显存。区分完整模型参数与实际调用模块参数；旧兼容/未调用模块不能被当作实际计算量，也不能静默从加载内存中删去。

## 输出等价优化与退出规则

先保留原参考实现的计时与诊断开销清单。若移除CoSDH重复decode/warp，必须保留由发送端本地置信度和late框产生的必要计算，优化版在同64帧与原参考预测、包内容/字节逐位一致后才能测量；任何有容差比较须单独解释，不自动当作逐位等价。InfoCom、SNN采取同样原则。计时中用内存收集器替代包文件写入，必须真实执行原codec并在计时外核对实际字节与SHA，不能用预缓存包代替发送计算。

只允许一轮有界的等价计时接入；若等价门不通过或实现尚未消除诊断冗余，保留参考结果并明确不可作速度主表，不用反复局部调参推迟主线。主要交付为各方法全部原始64×3计时、均值/中位/P95、实际通信、参数/显存及数值等价证据。MAC/AC只在算子覆盖可核查时另报，不能把普通CUDA卷积中的零值自动算作已节省的AC。GPU计时不能替代神经形态硬件能耗。

验收允许结果为SNN更慢或无优势；若只有模型存储较小，则只报告存储权衡。结合完整AP—bytes曲线决定下一轮机制实验，不能以删除codec、忽略发送端或改数据分辨率制造优势。


## 2026-10-26 16:11：共同计时核心已实现，真实adapter待接入

`opencood/tools/strong_ann_baselines/compute_harness.py`实现两种明确边界：CPU ready batch→CPU框，以及已在GPU的batch→CPU框；开始/结束同步，实际输出D2H包含在计时内，包哈希/参考核对在结束计时之后。warmup执行真实调用但不计入64帧统计；原始逐帧数据、重复编号、前后资源快照全部保留，不自动删除有干扰数据。回调必须实际执行消息选择/codec/接收，不允许参考包缓存替代真实计算。

5项CPU回归验证边界、D2H位置、哈希在计时外、同长度错误包/错帧拒绝、重复方向标签/非有限输出拒绝、warmup排除且保留资源干扰、固定64索引。Windows和181均通过。首次以`opencood.test.test_compute_harness`运行触发仓库原有`opencood/test.py`命名冲突及其外部数据路径错误；改用`unittest discover -s opencood/test -p test_compute_harness.py`正确发现测试，未修改数据或原测试模块。

代码7c4c6f7已推main/181 pull。仅为CPU可验证的计时基础设施，真实模型adapter、显存采集、同卡独占证据、阶段钩子与4帧GPU等价门尚未完成。先用已冻结官方InfoCom10%工作点开发adapter，正式主表仍等待适配选模完成；之后接入SNN/CoSDH。不能将合成adapter测试当成实际检测数值/性能证据。下一轮优先实现并运行真实4帧门，不重复扩充协议文档替代落地。


## 2026-10-26 17:10：真实InfoCom adapter已接入，修正输入重放后检查中

14dab83新增compute_infocom_adapter和显式compute-smoke-reference入口，复用既有官方模型严格加载、原生推理/后处理和真实WireAudit；计时中不调用finish写包文件，保留真实encode/decode和接收端使用解码消息。使用官方10%冻结源权重、既有全量参考的4个固定索引0/579/1158/1737及原包SHA；参考来自146已完成1738帧验证，压缩归档SHA=43cfa28f47bcbf7b662c135b6827444da8eda7eca97cd3adf6d23a2266c321e9。源权重和逐帧包未变。

181 GPU2五次准入均24252MiB/0%，smoke01在数值/包内容严格等价门停止，失败已保留。源码检查发现官方评估路径也调用numpy随机shuffle_points；直接随机访问四帧，且每次warmup重新加载/体素化，会改变随机数序列与体素内取点。这是计时输入一致性缺陷，不能放松预测或包的逐位要求来掩盖。

8bbf106改为按原seed和完整有序DataLoader重放预处理，只缓存四帧CPU输入；40次warmup/8次计时复用这些输入，但每次重新执行完整网络、消息选择、实际codec和接收，不缓存模型输出/消息。to_device递归创建新容器，CPU输入保留。已补CPU回归验证不再随机访问数据集、重复调用清空消息及错帧拒绝，连同原计时核心本机7项通过。

基于上述明确修正，smoke02独立目录/PID45902启动，GPU2重新五次准入通过，外层1800秒（包含完整输入重放），无自动重试；smoke01失败保留。当前处于preparing_reference_inputs，**尚未证明修正后四帧等价，也没有受控速度结果**。若仍失败，先输出差异定位，不能继续放松精度门或改模型。现有训练与SNN评估未改动。下一次读取smoke02的compute_summary/failure/status，不能重复启动。

证据：[原始失败](results/strong_ann_baselines_20261025/heartbeat_20261026_0910/compute/infocom_compute_smoke01/failure.json)、[修正后启动与理由](results/strong_ann_baselines_20261025/heartbeat_20261026_0910/compute/infocom_compute_smoke02.launch.json)、[门检查协议](results/strong_ann_baselines_20261025/heartbeat_20261026_0910/compute/infocom_compute_smoke02/compute_protocol.json)。此为有界工程门检查，不是增加研究训练臂；正常中期进展保持静默。


## 2026-10-26 18:12：InfoCom和方案二实际四帧等价门通过

InfoCom smoke02已完成：完整有序DataLoader预处理重放后，复用四帧CPU输入，每次重新执行真实网络、选择/codec与接收。两种计时边界共40次warmup和8次正式调用都与既有1738帧参考预测和包内容逐位一致，原smoke01失败保留。这验证了输入重放修正后的门，不放松数值门；仍不能用4帧时间或不同模型的历史日志宣布速度优势。

进一步实现de21ff7的Scheme-2 compute_snapshot_smoke。模型增加显式可选的真实包observer（仅eval+packet_roundtrip可用，默认无操作），将已生成的真实packet交给内存收集器，计时中关闭packet_directory磁盘导出；消息选择、实际压缩/解码、三维对齐、接收、BEV/头和NMS均保留。加载与完整2KiB复核相同的第6轮最佳检查点及config，除禁用导出目录外不改模型配置。PairedDAIRFusion评估仅在train时shuffle，四帧CPU输入可直接固定复用；不缓存模型或消息结果。

181 GPU2重新五次24252MiB/0%准入后，scheme2_compute_smoke01/PID12505已完成。相同索引0/579/1158/1737、两种边界、40次warmup/8次计时均通过完整预测和真实包SHA/长度等价；预算仍2048 B，未修改当前训练或8KiB评估进程。两个门的数据及资源快照已取回，均标smoke_only/no_full_AP_or_speed_claim。

下一步先为CoSDH接入同样的输入/真实包等价门，明确其审计重复decode/warp；再等待InfoCom最终选模，将冻结后的全部工作点接到共同64帧×3轮计时。参数/显存可独立报告，但正式速度主表必须满足共同边界与干扰记录，不由本轮四帧粗测提前下结论。多预算后续8KiB收尾也需提前检查原6小时监督上限，避免同类超时，不能热改当前模型或减少评估帧。

证据：[InfoCom真实门](results/strong_ann_baselines_20261025/heartbeat_20261026_1012/infocom_compute/compute_summary.json)、[方案二真实门](results/strong_ann_baselines_20261025/heartbeat_20261026_1012/scheme2_compute/compute_summary.json)、[方案二加载/配置/脚本摘要](results/strong_ann_baselines_20261025/heartbeat_20261026_1012/scheme2_compute/compute_protocol.json)。


## 2026-10-26 20:10触发后：CoSDH真实计算等价门已启动

fe5e597接入CoSDH真实双向消息计时。CoSDHWire.packets将原finish中的envelope/section拼接分离为无文件IO函数；原正式评估finish仍使用同一组包函数。新adapter在infer返回前完成真实组包，确保双向包头和拼接没有被排除在计时之外；SHA和参考对比放在计时后。warmup与计时复用按原seed303有序重放的四帧CPU输入，每次仍执行完整官方hybrid双前向、选择、编解码、对齐、接收与NMS，不缓存消息或模型结果。

CoSDH官方源b270b69fe133c0c1ca2044ad072cd77fc21863fb、原net_epoch11.pth/config以及全量参考的4帧预测/包从146独立归档至181；归档SHA=71bac97772034bb3de3df97ccd9e4dd81b2034acad5ec7bbc83fc7ebedbc26a8，部署目录cosdh_compute_source01，未覆盖其他项目。运行时借用同181环境已经可用的Python3.7编译扩展。源目录为归档快照无.git，provenance中repo_commit为空，原提交与归档摘要在deployment和launch记录，不把空字段当作来源缺失。

原GPU4在启动前出现其他负载，准入拒绝且没有启动模型；改GPU5五次采样均24252MiB空闲/0%，内存比例上限0.25。smoke01/PID38875已启动，外层1800秒；严格权重键/形状审计通过，进入完整有序数据预处理重放。当前仅启动并通过加载，尚未证明四帧预测/包等价。相关10项CPU测试在本机、181均通过，覆盖组包与原文件导出的字节一致性、计时内组包及禁止旧消息复用。

该门明确保留参考codec中的重复诊断decode/warp，标eligible_for_main_speed_table=false。即使通过，也只能证明计算入口等价，不能直接作优化后CoSDH速度主表；需按预登记有界地消除冗余并再次等价验证，或如实保留不可公平计时的限制。不将四帧结果当作完整64帧×3轮成本，也不将真实GPU稀疏零值直接折算为神经形态AC节能。现有SNN多预算训练未改动。

证据：[准入、命令、来源和PID](results/strong_ann_baselines_20261025/heartbeat_20261026_1210/cosdh_compute_smoke01.launch.json)。下轮检查cosdh_compute_smoke01的compute_summary/failure/status，不重复启动。


## 2026-10-26 21:05触发：CoSDH等价门通过，登记共同64帧测量

CoSDH smoke01两种边界、40次warmup及8次正式调用已完成；4个固定帧的预测与实际双向包内容均与既有全量参考逐位一致。真实组包位于infer返回前，文件导出不计入。归档SHA=7ffff2976c4723663a99539342c784c6a51d3a4ab922fac8c44b0b94c82fbae6，已取回并镜像。严格加载无缺键、多键、形状差异。仅证明计时入口数值等价，未证明速度优势。

本轮在原共同计算协议内执行64帧×3轮测量，固定四个工作点顺序：方案二固定2KiB最佳第6轮、InfoCom适配最佳第8轮1%、InfoCom官方10%、CoSDH官方allpairs参考。第0轮按此顺序，第1/2轮分别左移1/2位，全部在同一181 RTX3090物理卡上串行运行，每方法每轮每种边界先20次真实warmup再64次测量。每次重新加载冻结检查点，三轮使用相同64个ID和输入重放种子；原始逐帧时间、包字节、资源快照、驱动/CPU/Torch环境、显存完整保留。不做新的64帧AP选模，不按快慢选择epoch或删除异常值。

CoSDH仍保留重复诊断解码/warp，本次仅作为不可用于公平速度排名的参考成本行，eligible_for_speed_table=false。其他方法也执行真实codec；不得用CoSDH含冗余的时间宣称SNN胜出。去除冗余需要后续有界等价优化，本轮不为等待该工程而阻塞其他已满足边界的方法。InfoCom官方和适配分开列，不以低预算模型高比例退化替代官方高比例工作点。

64帧新增参考包与预测由146既有全量结果提取，无重算或改变GT，归档SHA=0ce86108d9e9efd48ab1f1ddc1af22beb662a5fe84c146a28044f463bd749ffc。方案二和适配InfoCom参考已在181。每次正式调用严格比对真实包和输出，任何差异立即停止该方法，保留失败且不自动重试，其余独立方法继续。每子进程超时2700秒并仅终止其自身进程组；最坏12次总窗口9小时。全部模型保持25%显存分配上限，启动前检查GPU资源。执行从main导出的只读代码快照，后续文档同步不影响运行代码。

代码将原四帧门扩展为显式4/64帧选择和0/1/2重复编号，默认仍为四帧；11项相关CPU测试通过。新监督器只执行预登记四方法三轮，不选择最好重复，不自动重启失败方法。当前多预算SNN第3轮751/2339，正常运行，未新增AP；保持原训练预算及方案三暂停。下一步取回三轮原始结果、核对共同GPU/干扰和输出等价后再形成成本表。

证据：[CoSDH等价门结果](results/strong_ann_baselines_20261025/heartbeat_20261026_1305/cosdh_gate/compute_summary.json)。


启动记录补充：a84426d/c7b1c9a已推main并同步181/146。启动前GPU4–7被其他用户新任务占用，GPU5利用率约91–99%，准入拒绝，未在高争用卡上启动模型。为避免把争用延迟归因于网络，compute64_suite01/PID17280进入资源队列；当前没有新的64帧测量结果。执行快照compute64_source02固定c7b1c9a，11项测试在181通过。每300秒检查固定GPU5，无计算进程且空闲显存≥16GB、利用率≤20%的5次连续采样后开始；最多等待6小时，每方法开始前再次检查。此更严格资源要求仅用于受控成本测量，不撤销用户允许训练共享低利用率GPU的授权，也不停止任何现有训练。

队列当前waiting_for_gpu，第一项scheme2_2k/repeat0；GPU UUID为GPU-1fa76848-4737-2fdc-5205-5314117e890f。快照和plan02已冻结，source01/plan01因资源拒绝未执行，保留其来源。后续读取compute64_suite01/status.json、results_so_far.json、summary.json与failure.json；若resource_timeout，只记录资源阻塞并重新核实可用卡，不能误报测量完成或自动选择更快样本。归档SHA=856a6a07b9a106b24dc0339fdd47d6484a7997b5a029e36abdb2e617a0446812。

证据：[固定四方法三轮计划](results/strong_ann_baselines_20261025/heartbeat_20261026_1305/compute64_queue/compute64_plan02.json)、[队列PID](results/strong_ann_baselines_20261025/heartbeat_20261026_1305/compute64_queue/compute64_suite01.launch.json)、[资源等待状态](results/strong_ann_baselines_20261025/heartbeat_20261026_1305/compute64_queue/status.json)。


## 2026-10-26 22:02触发：64帧门失败已定位到单帧消息，检测输出无差异

compute64_suite01首次方案二repeat0曾通过GPU准入并启动，但在实际输出/包的严格等价门停止，returncode=1。失败及原协议保留，监督器跳过该方法剩余重复，不自动重试；其余ANN方法仍排队。没有完整SNN64帧计时，不能从局部调用生成速度表。此前四帧通过不保证64帧通过。

3ab2a28增加显式--diagnose，不改变正式门、不放宽任何容差。在相同检查点/config/64个ID上运行两遍不计时诊断，导出所有不一致帧的数组、实际包与指纹。依据用户允许共享低负载GPU的授权，使用181 GPU1，五次空闲16789MiB、利用率64/14/35/57/48%（平均43.6%），25%显存上限，900秒上限；GPU2先前准入失败未启动。共享诊断不提供速度证据。PID45595任务已完成。

128次调用中，全部检测框与分数同原始全量参考逐位一致。仅index579/frame004276两次消息不一致，且两次诊断自身逐位一致：实际2046 B，原参考2047 B。解析实际包与原包后，身份、坐标系、位姿、通道基及尺度全部相同；差异来自两个单位计数，实际2083个、原2085个。两个差异坐标为[batch0,channel3,z2,y16,x110]和[batch0,channel16,z2,y16,x110]，实际均0、参考均1。故不是纯压缩编码变化或元数据差异；发送端输入、数值计算或事件前缀选择的具体来源尚未确定。不可将这些证据直接称为浮点误差修复或算法bug已解决。

结论边界：本次诊断没有发现检测AP输出变化；保留原1738帧AP结果及其原实际通信，不能将其包字节悄悄替换成当前诊断值，也不能因此宣布整个1738帧的发送过程跨运行完全稳定。正式通信逐位门仍失败，SNN速度行继续不可用，不修改阈值凑通过。下一步有界定位该帧的输入/发送计数/排序前缀，并在原因明确后才决定修复与重新复核；不重训模型，不把通信差异掩盖成检测改善。继续原多预算训练（第3轮1051/2339，暂无新AP）及ANN计算队列，不恢复方案三。

诊断记录与序列化回归测试加入后，本机12项计算相关测试通过；新增测试证明仅分数变化也被保留、不能被等价通过。main与181/146同步；运行中的原计算队列使用既有快照，不热改进程。证据：[失败](results/strong_ann_baselines_20261025/heartbeat_20261026_1402/failed64/failure.json)、[128次诊断](results/strong_ann_baselines_20261025/heartbeat_20261026_1402/diagnostic/diagnostic_summary.json)、[解码计数差异](results/strong_ann_baselines_20261025/heartbeat_20261026_1402/diagnostic/packet_difference.json)、[实际与参考指纹](results/strong_ann_baselines_20261025/heartbeat_20261026_1402/diagnostic/mismatch_r0_i0579/fingerprints.json)。回传归档SHA=22de64eec650bc71eb269053eaad4a55435b09534e40903b438c421a477e3042。


## 2026-10-26 23:05触发：InfoCom随机平局重放缺陷与SNN主干差异分开定位

原compute64_suite01的InfoCom适配1%和官方10%首轮均在严格等价门失败，未自动重跑。两组固定64帧各两遍的不计时诊断已完成：适配1%共127/128次与参考不一致，64帧跨遍也均不稳定，包含框/分数变化；官方10%只有index937/frame011530两次消息包不一致，所有检测框/分数仍逐位一致。不得将前者归结为小浮点误差，更不能放宽数值门后称通过。

查明官方InfoCom extensions/utils.py的topk_mask在阈值平局时使用torch.randperm，eval也执行。原完整评估按1738帧顺序消耗CUDA RNG，且第0帧3次warmup后恢复random/NumPy/Torch/CUDA随机状态；旧计时入口只重放预处理、跳过中间模型调用，额外warmup又改变CUDA随机序列。因此过去四帧通过不足以覆盖有大量平局的1%适配模型。原完整AP仍是固定seed303及原顺序下的结果，独立CPU复算没有失效；不能把该单次结果说成与随机平局选择无关的稳健性能。

72efd28修复计时入口：按原顺序执行完整输入和模型重放，精确复制第0帧warmup及随后RNG恢复，在64个选中帧前保存CPU/CUDA RNG；重放时就将这些帧的实际包和输出同原参考逐位核对。正式warmup/计时每次在计时外恢复该帧原RNG，再执行真实网络、排序/随机平局选择、codec和接收；只缓存CPU输入/RNG，不缓存输出或消息，不改变官方平局规则/权重/阈值。13项相关测试在本机及181通过，严格GPU门尚待运行，不能提前声称已通过。

仅对这两种InfoCom工作点进行有理由的工程修复复核，仍固定64帧×3轮和原模型/种子，不新增训练。compute64_source03固定72efd28；compute64_infocom_rng02由PID48209等待原compute64_suite01进入终态后，再在同一GPU5检查资源并启动，避免两个监督器争抢同卡。先前rng01监督器PID47419在waiting_for_gpu且无active_child时终止，未执行任何模型，保存superseded_before_model_start记录。新路径每轮先完整RNG重放，任何新差异保留失败且不自动重复；本次有界修复不开展额外ANN参数搜索。

SNN另有独立证据：同GPU6、同权重的4帧/64帧两遍trace中，4帧全部通过，64帧仍仅原004276差异；同一运行内两遍一致。投影前的[1,128,3,50,126]整数体素表征有且仅有一个值不同：[0,100,2,16,110]由3变2，导致同址32通道投影驱动改变（最大绝对差0.4037489891），再少发两个脉冲。差异已出现在3D主干末端，不能再笼统归因于压缩器或仅投影浮点误差。根因仍需区分输入/主干数值及量化边界；未修改模型或容差。

为完成这一次有界定位，GPU6/PID48850按顺序执行scheme2_threshold4_01与scheme2_threshold64_01：额外记录该帧输入体素均值/坐标及最后Count4前目标位置浮点值，所有钩子只观测、不替换张量，仍不计时。两项均1500秒上限、冻结同一代码/检查点；结果出来后先决定最小修复或明确可复现性限制，不继续无限逐层诊断。主训练仍第3轮1351/2339，暂无新验证AP；不因计时问题重启训练或恢复方案三。

CoSDH参考首轮64帧、两种计时边界的128次正式调用已全部通过原预测/双向包逐位等价；原三轮队列继续。仍含重复审计decode/warp，不能用它的时间宣称SNN或InfoCom速度优势。阶段成本结论继续未完成。

证据：[InfoCom适配诊断](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/infocom_adapt1pct_diagnostic01/diagnostic_summary.json)、[官方模型诊断](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/infocom_official10pct_diagnostic01/diagnostic_summary.json)、[SNN输入投影比较](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/scheme2_sender_trace_comparison.json)、[修复后双方法计划](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/compute64_infocom_rng_plan01.json)、[串行依赖启动](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/compute64_infocom_rng02.launch.json)、[量化前观测启动](results/strong_ann_baselines_20261025/heartbeat_20261026_1505/scheme2_threshold_pair.launch.json)。回传归档SHA=4f9892afaf00fe84bfa8a8c6ec3f88c5b3f9296da11babf8190e6a76e337d167。
