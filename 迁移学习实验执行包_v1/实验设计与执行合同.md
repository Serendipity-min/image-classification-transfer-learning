# 图像分类迁移学习实验设计与 Codex 执行合同

版本：1.0  
协议编号：TL-CLS-20261020-v1.0  
制定日期：2026-10-20  
适用阶段：预实验复核、正式实验设置、Codex 实施与结果回交  
当前状态：设计已制定，服务器运行与实验结果尚未验收

## 一 研究裁定与本轮目标

本项目可继续推进。原稿提供了值得验证的现象，但实验矩阵、模型冻结、测试集使用和失败归因存在缺口。新实验首先建立可信的分类比较，再补充数据预算和预训练来源对照。

保留原稿的研究对象：SVHN、CIFAR-100，ResNet-50、ViT-B/16，以及从头训练、全量微调、严格线性探测、Adapter 微调四种策略。正式主矩阵为 **16 种配置，每种预先固定 5 个训练种子，共 80 次正式训练**。前置工作包括 16 次短程验收与 48 次有限学习率试验。

本研究的主要产出是明确资源约束下的可复现实证证据和方法选择依据。四种已有方法的组合比较不能自动构成新算法贡献，也不能预先保证发表结果。原稿报告的准确率属于待核验的历史观察，不设为必须达到的验收门槛。

本次交付没有登录服务器，没有运行训练，也没有获得新的准确率、耗时或显存结果。所有运行数量为计划，参数数量为根据明示结构的算术预期；实际值由 Codex 输出原始收据。

### 1.1 分工

| 角色 | 职责 | 决策范围 |
|---|---|---|
| 用户 | 项目总负责人，审核与监督，调整研究方向和总预算 | 研究目标、重要范围变化、预算增加、最终结论与投稿 |
| ChatGPT | 实验设计、科学口径、阶段审查、结果分析 | 制定协议；核查数据与实现；解释结果及证据强度；提出必要补实验 |
| Codex | 工程实施与实验执行 | 实现训练与评价程序；技术验收；按冻结配置执行；保存完整证据并回交 |

常规技术修复可以在当前阶段内处理，不必反复征求安装依赖或整理目录的许可。影响科学比较的修改，如换权重、换划分、改精度、改训练预算、改 Adapter 拓扑，必须产生协议修订及影响说明，不能静默混入原组。

### 1.2 输入证据

原始文档：资源受限场景下图像分类迁移学习策略的对比与实证研究.docx。

SHA-256：

~~~text
41d98ea1d77176523a9e5881c4168a0aa7526f39f2ab889579b6b3c94464af89
~~~

服务器规格由用户提供，尚待 Codex 在运行时核验。没有收到原始实验代码、训练日志、checkpoint、准确软件版本与完整启动命令，因此目前不能宣称已精确复现旧结果。缺失旧代码不阻止建立本轮规范实验。

## 二 原稿审查与必须修正的问题

| 原稿中的问题 | 证据位置或具体内容 | 本轮处理 |
|---|---|---|
| 四策略矩阵不完整 | 第 4.5、6.2 节：SVHN 的 ViT-LP 实为 Partial FT；CIFAR-100 缺两个 Adapter | 重做完整 16 配置；错误 Partial FT 不计作 LP |
| 单次运行无法支持小差距结论 | 所有实验 seed=42；ViT-CIFAR LP 比 FT 高 0.46 个百分点 | 主矩阵固定 5 个种子；报告逐种子结果、配对差值与区间 |
| 测试集参与选模 | 第 4.3 节 Best Top-1，表 5 逐 epoch Test Top-1 | 从官方训练集划验证集；仅用验证集选超参数和 checkpoint |
| 比较同时改变了训练预算 | Scratch/FT/LP/Adapter 为 100/30/50/50 epochs，batch 不同 | 新主矩阵统一 50 epochs 与有效 batch；旧配方单列 |
| 纯架构与预训练规模无法识别 | R50 用 IN1K，ViT 用 IN21K，配方、参数规模也不同 | 结论限定为具体模型与权重组合；补同架构 IN1K 对照 |
| 两种 Adapter 不是相同容量或相同模块 | ViT 为 Linear+GELU；R50 为 Conv+BN+ReLU；训练占比约 4% 与 30% | 明确各自结构和分母，避免将差异单独归因于 CNN/ViT |
| 初始化说明不成立 | 第 3.4 节认为残差连接自动保证初始输出不变 | 显式零初始化输出分支，并验收初始输出 |
| SVHN 增强需要纠正 | 第 4.4 节所有数据统一 RandomHorizontalFlip | SVHN 取消水平翻转；CIFAR-100 保留 |
| 输入预处理不明确 | 目标数据集均值方差未提供数值；模型默认配置不同 | 使用明确的几何变换及权重推荐标准化，固定到配置 |
| NaN 被过度解释 | 将单次 NaN 解释为 ViT 不可从头训练、Adapter 与架构不兼容 | 先排查实现、精度、初始化和更新；因果解释须有干预证据 |
| 参数效率被当作整体资源效率 | 用可训练参数量推断显存、训练和部署成本 | 实测时间、显存、任务增量存储与推理成本 |
| 引用需要核验 | 原稿参考文献 [4] 的作者与发表信息值得核对 | 按 DOI、作者主页或论文原页重建条目，不为凑数量添加引用 |

覆盖率的准确表述是：16 个目标单元中，13 个以正确策略被尝试，其中 2 个数值失败；正确策略且成功完成的单元为 **11/16**。

### 2.1 数字复核

| 原稿比较 | 复核结果 | 解释边界 |
|---|---:|---|
| ViT-CIFAR LP − FT | 88.65 − 88.19 = 0.46 个百分点 | 单种子不能证明 LP 更优 |
| R50-SVHN Adapter − FT | 96.25 − 92.46 = 3.79 个百分点 | 仍需排查预算、增强、学习率与实现 |
| R50-SVHN Adapter − Scratch | 0.39 个百分点 | 不宜用夸张语言描述领先 |
| R50-CIFAR FT − Scratch | 13.06 个百分点 | 是旧配方下的观察 |
| ViT-SVHN FT − Scratch | 8.81 个百分点 | 不代表所有任务中的预训练增益 |
| ViT-CIFAR LP 耗时变化 | 29:17 → 21:03，减少 28.116% | 对应加速比 1.391 倍；原硬件与计时边界未核验 |
| ViT-CIFAR LP 参数占比 | 76,900 / 85,875,556 = 0.08955% | 约为 FT 的 1/1117，原稿 1/1073 来自舍入后相除 |

文稿中的主要参数数字能在明确实现假设下复算。审计重点在方法身份和结论强度，不以数字偏高为由推定结果不真实。

## 三 研究问题与结论范围

### 3.1 主问题

**RQ1 适配效果与代价。** 在相同数据、评价口径、有限选参和明确训练预算下，LP、Adapter 相对 FT 的精度变化是多少，节约了哪些资源？

**RQ2 结果是否依赖具体配置。** 上述差异在两个数据集和两个指定模型组合中是否一致？报告交互现象，不把多个共同变化的因素拆成未经识别的因果结论。

**RQ3 数值失败的条件。** 历史 NaN 能否被重建？异常首先出现在输入、前向、梯度还是参数更新？单因素干预能否改变失败？

**RQ4 资源预算变化。** 在同一训练轨迹的共同时间上限、较少下游训练样本以及不同预训练来源下，方法排序是否改变？

### 3.2 预先指定的主要比较

1. CIFAR-100、ViT-B/16 IN21K：LP 对 FT，检验原稿的小幅精度优势和成本优势。
2. SVHN、R50 IN1K：Adapter 对 FT，检验原稿约 3.79 个百分点的差异。

其余配置完整报告，作为次要比较与稳定性证据，不在看到结果后改成“唯一主实验”。

### 3.3 可以支持与不能支持的结论

可以支持：“在本数据、权重、候选超参数与预算下，方法 A 与 B 的平均差值及成本差异为多少。”

尚不能支持：“ViT 必须使用预训练”“Adapter 天生不适合 ViT”“全部差异由预训练数据规模引起”“全量微调一般最稳定”“少量训练参数必然减少推理模型大小”。

DeiT 已展示在合适训练配方下，仅使用 ImageNet 训练视觉 Transformer 的可行性。这足以反驳“Transformer 没有外部预训练就不能训练”的泛化说法，但不保证本项目的 ViT-B/16 在 CIFAR-100 和 50 epochs 内达到高精度。[S1]

现有工作也已系统讨论视觉 PEFT 的公平比较，以及参数效率与时间、内存效率的区别。本文应强调透明协议、完整成本和适用条件，而不声称首次比较经典方法。[S2–S4]

## 四 核心实验矩阵

| 数据集 | 骨干与权重 | Scratch | Full FT | LP | Adapter |
|---|---|---:|---:|---:|---:|
| SVHN | ResNet-50 IN1K | 5 | 5 | 5 | 5 |
| SVHN | ViT-B/16 IN21K | 5 | 5 | 5 | 5 |
| CIFAR-100 | ResNet-50 IN1K | 5 | 5 | 5 | 5 |
| CIFAR-100 | ViT-B/16 IN21K | 5 | 5 | 5 | 5 |

表内数字为正式运行次数。合计 80 次，每次 50 epochs。Scratch 不加载任何预训练权重；表中权重标记对 Scratch 仅表示与同组共享架构和输入标准化。

训练种子固定为：

~~~text
17, 42, 2026, 3407, 65537
~~~

划分种子为 20261020；短程验收种子为 314159；学习率选择种子为 123。调参运行不计入正式五种子统计。

相同 dataset、backbone、seed 的四策略使用相同的数据顺序与增强随机序列，并独立生成一致的分类头初始参数。模型初始化、Adapter 初始化、sampler 和 augmentation 使用分别可恢复的 RNG 流，避免插入新模块后消耗随机数导致数据配对失效。增强随机数由 training_seed、epoch、带官方 split 的 sample_id、augmentation_view_id 的固定 SHA-256 派生（取低 64 位作为局部 RNG seed），不用 Python 进程随机化的 hash()。这样不依赖 persistent worker 的隐含 RNG 进度。各 epoch 的 sampler 顺序另存或按已冻结算法确定性重建。

## 五 数据协议

### 5.1 数据来源与划分

| 数据集 | 官方训练数 | 本轮训练 | 本轮验证 | 官方测试 |
|---|---:|---:|---:|---:|
| CIFAR-100 | 50,000 | 45,000 | 5,000 | 10,000 |
| SVHN cropped 32×32 | 73,257 | 65,931 | 7,326 | 26,032 |

CIFAR-100 使用 100 个细类标签，每类官方训练样本 500 张。每类固定划出 50 张验证、450 张训练。[S5]

SVHN 使用 cropped32 的 train/test；主实验不使用 extra。验证集大小精确为 7,326，按类别保持自然比例。实现为先分配每类的下取整数配额，再按小数余数从大到小补足，余数并列按类别 ID；各类内部用固定 RNG 打乱。保存实际索引后，所有方法复用同一文件。

TorchVision 已把 SVHN 原始标签 10 映射为 0，使用其 Dataset 后不能再整体减 1。加载 MAT 文件需验证 SciPy 可用。[S6]

### 5.2 完整性和隔离

保存原始文件的官方 MD5 与本地 SHA-256、样本数、类别数、逐类计数和 split manifest。CIFAR-100 Python 压缩包 MD5 为 `eb9058c3a382ffc7106e4002c42a8d85`；SVHN train/test 的 MD5 分别为 `e26dedcc434d2e4c54c9b2d4a06d8373` 和 `eb5a983be6a315427106f1b164d9cef3`。[S5,S6]

样本 ID 必须包含 dataset 和官方 split 命名空间，不能因为 train/test 各自都有 index=0 就误判重叠。断言本轮 train/val 的官方训练索引不重叠，并验证测试来源为独立官方 split。对原始像素精确重复进行审计并记录，不根据结果悄悄删样本或更改官方测试集。

原研究已经查看过公开测试集。本轮保证的是新的配置冻结前不再次使用测试成绩调参，不能把它包装成整个研究历史上从未见过的测试集。

P0 可以下载、校验测试文件并核对结构，训练与选参程序不得构建 test evaluator。P3 后由单独评价入口对冻结模型执行测试。测试标签不得用于标准化统计、增强选择、超参数搜索、异常定位或提前停止。

### 5.3 图像变换

| 阶段 | CIFAR-100 | SVHN |
|---|---|---|
| 训练几何变换 | Resize(256) → RandomCrop(224) → HFlip(p=0.5) | Resize(256) → RandomCrop(224)，不翻转 |
| 验证与测试 | Resize(256) → CenterCrop(224) | Resize(256) → CenterCrop(224) |
| 插值 | BICUBIC，antialias=True | BICUBIC，antialias=True |
| 通道与张量 | RGB，ToTensor，输入 3×224×224 | RGB，ToTensor，输入 3×224×224 |

保持原稿的基本裁剪流程，减少不必要的设计扩张。主实验不加入 Mixup、CutMix、RandAugment、随机擦除、标签平滑或测试时增强。

标准化按骨干对应权重配置固定：

| 骨干 | Mean | Std |
|---|---|---|
| R50 A1 | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| ViT AugReg | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 |

每个骨干的四种策略共享该标准化，Scratch 也相同。跨骨干标准化不同是模型管线的一部分，不能据此宣称完全隔离了架构变量。R50 A1 模型卡默认 test288，本项目明确覆盖为 224；不能直接使用默认 transform 导致不同输入尺寸。[S7,S8]

## 六 模型和策略的精确定义

### 6.1 权重身份

| 简写 | 精确模型标识 | 来源 |
|---|---|---|
| r50 | `resnet50.a1_in1k` | timm，ResNet Strikes Back A1，ImageNet-1K |
| vitb16 | `vit_base_patch16_224.augreg_in21k` | 原作者 JAX AugReg IN21K 权重，timm 移植 |

`a1_in1k` 是 timm 权重身份，不能用 TorchVision DEFAULT 权重静默替代。ViT IN21K 原分类头为 21,843 类；替换下游分类头后参数量变化，因此不能直接把原模型卡的 102.6M 当作下游模型参数量。[S7,S8]

P0 保存 model ID、仓库 revision、权重文件名、SHA-256、预训练来源、原始分类数、目标分类数、timm 版本和模型构造参数。不能凭名称相似换成 `in21k_ft_in1k`。

分类头均为纯线性层，带 bias；R50 为 Linear(2048,C)，ViT 为 Linear(768,C)。新头权重使用截断正态分布 std=0.02，bias=0；记录实际张量 hash。同组同种子四策略的头初始化一致。

### 6.2 四种策略

| 策略 | 初始化 | 可更新参数 | 主干训练状态 |
|---|---|---|---|
| Scratch | 主干随机初始化，新头 | 全模型 | train，BN 正常更新 |
| Full FT | 预训练主干，新头 | 全模型 | train，BN 正常更新 |
| LP | 预训练主干，新头 | 仅纯线性分类头 | backbone eval，参数及 buffers 固定 |
| Adapter | 预训练主干，新头及新增 Adapter | 仅 Adapter 与头 | 原主干参数固定，原 BN eval；新增 Adapter BN train |

使用明确模块对象或完整白名单解冻参数，禁止用参数名中包含 `fc` 的规则。优化器中的参数 ID 集合必须与白名单精确相等，不得遗漏可训练参数或混入冻结参数。

LP 可以用 `no_grad` 计算主干特征。Adapter 位于网络内部，其梯度需要穿过后续冻结层；不能把整个骨干包在 `no_grad` 或 `inference_mode` 中。`eval()` 不等于关闭梯度。[S9]

### 6.3 主实验 Adapter

主实验名为 `canonical_adapter`，保持原文插入位置与瓶颈比例，补齐原稿没有冻结的实现细节。它不是 AdaptFormer。

**ViT：**12 个 block，每个 block 输出 y 替换为：

\[
y + W_{\mathrm{up}}\operatorname{GELU}(W_{\mathrm{down}}y+b_{\mathrm{down}})+b_{\mathrm{up}}.
\]

通道为 768→192→768，两层 Linear 均带 bias。down 权重 Kaiming uniform，参数 a=√5；down bias=0；up 权重和 bias=0。不开额外 LayerNorm、dropout、可学习 scale。不能同时把 down 和 up 权重全部置零。

**R50：**四个 stage 的 3、4、6、3 个 Bottleneck 输出后各加：

~~~text
Conv1×1(c,c/4,bias=False)
→ BN(c/4)
→ ReLU
→ Conv1×1(c/4,c,bias=False)
→ BN(c)
→ 与 Bottleneck 输出相加
~~~

卷积权重 Kaiming normal，mode=fan_out、nonlinearity=relu；首个新 BN 的 gamma=1、beta=0；末个新 BN 的 gamma=0、beta=0。BN 使用 eps=1e-5、momentum=0.1、track_running_stats=True。不能同时把末卷积与末 BN gamma 全部置零。

验证初始恒等时，两模型均 eval、使用同一头和固定 FP32 输入；加模块前后输出需在约定数值容差内一致。零初始化输出分支后，第一步 down 投影梯度为零可以是正常现象；应检查计算图连通、梯度有限，并在 2–3 次更新后确认前部 Adapter 的有效更新，不能要求第一步所有梯度均非零。

AdaptFormer 官方模块使用 FFN 并行分支、ReLU，并有输出零初始化、缩放等明确设置。后续若引入该强对照，必须单独命名和登记，不能静默替代原文的 block 后 GELU Adapter。[S10]

### 6.4 参数数量验收预期

以下整数以本节结构、纯线性分类头及模型无额外参数为前提。BN running mean/variance 等 buffers 不计入参数数量，但须计入增量权重存储。

| 模型与策略 | 类别数 | 可训练参数 | 完整模型参数 | 可训练占比 |
|---|---:|---:|---:|---:|
| R50 LP | 10 | 20,490 | 23,528,522 | 0.08709% |
| R50 LP | 100 | 204,900 | 23,712,932 | 0.86409% |
| ViT LP | 10 | 7,690 | 85,806,346 | 0.00896% |
| ViT LP | 100 | 76,900 | 85,875,556 | 0.08955% |
| R50 Adapter | 10 | 10,118,026 | 33,626,058 | 30.08984% |
| R50 Adapter | 100 | 10,302,436 | 33,810,468 | 30.47114% |
| ViT Adapter | 10 | 3,558,154 | 89,356,810 | 3.98196% |
| ViT Adapter | 100 | 3,627,364 | 89,426,020 | 4.05627% |

R50 新增 Adapter 参数共 10,097,536；ViT 共 3,550,464。Scratch 和 FT 可训练参数等于对应未插 Adapter 模型的全部参数。

每个 run 同时记录 `base_params`、`added_params`、`head_params`、`total_params`、`trainable_params`、`trainable_ratio`。模型实现与预期数值不符时先给差异明细，不硬改统计数字。

## 七 训练与有限选参协议

### 7.1 共同设置

| 项目 | 主实验设置 |
|---|---|
| 训练预算 | 全策略 50 epochs，正常运行不提前停止 |
| 物理 batch | 32 |
| 梯度累积 | 4；常规有效 batch=128 |
| 验证/测试 batch | 128；末 batch 保留全部样本 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| Weight decay | 1e-4，用于适用的权重矩阵/卷积；bias、Norm 及模型声明的 no_weight_decay 参数不衰减 |
| 损失 | CrossEntropy，label_smoothing=0，按实际样本数归一化 |
| 学习率调度 | 按 optimizer update 更新；从峰值的 1% 线性 warmup 5 epochs 后 cosine 至峰值的 1% |
| 梯度裁剪 | 全部可训练参数整体 global norm，max_norm=1.0 |
| 精度 | FP32 模型/优化器参数；BF16 autocast；logits 转 FP32 计算 loss；不使用 GradScaler |
| 主干正则化 | dropout、attention dropout、drop path 均显式设 0 |
| 加速选项 | 不启用 torch.compile、激活重计算、LP 特征缓存 |
| 重现设置 | TF32 关闭，cudnn.benchmark=False，记录确定性算法设置及支持情况 |
| 验证频率 | 每个完整 epoch 一次 |
| checkpoint 选择 | 最高 val Top-1；精确并列取较早 epoch |

BF16 是否可用由 P0/P1 实测确认。若不满足，则在正式实验开始前统一修订精度，不能只替换失败方法。FP16 的历史诊断与主实验分开。

训练不丢弃最后一个不完整有效 batch。梯度累积必须按本窗口真实样本数加权；最后不足 128 的窗口不能机械除以 4，避免梯度被错误缩小。报告真实 optimizer updates 与处理样本数。梯度累积不会让 R50 BN 的统计等价于物理 batch128；本文 BN 的物理 batch 是 32。

复现控制包括 Python/NumPy/torch/CUDA RNG、sampler、worker 与增强状态。P0 在固定版本和设备上测试确定性模式；遇到不支持的算子需记录并统一解决。不能承诺跨驱动、跨硬件逐位相同。[S11]

### 7.2 学习率选择

| 策略 | 3 个候选峰值学习率 | 分类头学习率 |
|---|---|---|
| Scratch | 3e-4, 1e-3, 3e-3 | 与主干相同 |
| Full FT | 1e-5, 3e-5, 1e-4 | 主干的 10 倍 |
| LP | 1e-3, 3e-3, 1e-2 | 该值即头学习率 |
| Adapter | 1e-4, 3e-4, 1e-3 | 与 Adapter 相同 |

FT 使用分组学习率是为区分预训练主干与随机新头的适应速度；该规则预先固定，不额外搜索倍数。

每个 dataset×backbone×strategy 单元使用调参 seed=123，跑 3 个候选、每个 15 epochs，总计 48 次。调参使用 50 epoch 的调度总时长，只运行前 15 epochs，warmup 仍为 5 epochs，使其成为正式配方的早期选择试验。

按第 13–15 epoch 的验证 Top-1 平均值选择候选；同分取较小学习率。候选必须完整结束且数值有效，否则标为不可选；如该单元所有候选失败，进入诊断，不自动替换策略。选参结果及全部失败、耗时、候选表写入 `selected_hparams.json`。

该设计控制候选数和训练暴露量，不保证各方法调参消耗相同 GPU 时间，也不保证找到全局最优配方。必须同时报告调参成本。15 epoch 早期选择可能偏向较快收敛的候选，这属于本研究的有限搜索条件。

主实验从相应初始化重新训练，不能把调参 checkpoint 延续成正式种子。正式 50 epoch 的结果是该预算下的表现，不能据此断定 Scratch 在 100 或 300 epochs 下也不可能改善。

## 八 资源约束与双卡执行

### 8.1 使用服务器的口径

每次训练占用一张 3090，最多同时两个训练进程。两张 24 GiB 卡用于并行不同 run；不默认视作单任务可用连续 48 GiB。本项目不需要 DDP。

按容器的 16 vCPU 与 64 GiB 内存配额分配资源，不按宿主机 80 逻辑线程、503 GiB 内存配置 DataLoader。初始每 run 4 workers，pin_memory=True、persistent_workers=True，OMP/MKL 线程数各为 1。

保留用户已工作的 Python 3.12.3、torch 2.5.1+cu124、torchvision 0.20.1+cu124。官方提供该 torch/torchvision/CUDA wheel 组合。独立项目环境中补充缺失依赖；timm 1.0.12 为初始固定候选，P0 构建与权重加载通过后锁定所有版本，不升级全局 CUDA/驱动。[S12,S13]

`df` 展示的 14 TB 剩余可能是共享挂载视图，不能当作租户已确认可用配额，也不能把两个挂载路径容量相加。建议项目先设置 100 GiB 存储规划上限，并在 P0 根据 checkpoint 数量计算；需要增长时给出明细。原始数据与基础权重只存一份，不按每个 run 复制。

### 8.2 计时口径

每个 run 是新进程，并记录物理 GPU UUID、PCIe ID、实际电源限制、温度、时钟、CPU 与其他占用。策略与种子应均衡分配到两张卡，避免方法与固定 GPU 绑定。

本研究核算下游适配成本，把已发布权重的上游预训练视为既有成本，不计入本次 GPU-hours。图表必须注明该范围；不能据此推断包含预训练的整个生命周期总成本。

保留以下独立字段：

1. 数据/权重下载与环境准备耗时，项目级单列。
2. `train_seconds`：训练循环耗时。
3. `validation_seconds`：验证耗时。
4. `checkpoint_seconds`：存盘耗时。
5. `fit_wall_seconds`：数据与权重在本地后，启动训练到最终保存结果，包含本次初始化、训练、验证和检查点。
6. `test_seconds`、推理基准和可视化耗时，另计。
7. `gpu_reserved_seconds`：run 累计占用 GPU 的时间；失败和重跑也计入项目成本。

精确 GPU 计时采用同步或 CUDA events，避免异步执行造成错误耗时。保留单卡独占的固定工作量基准，也保留真实双任务并行时的端到端耗时。两者名称分开，不能把推算时间写成实测。

双卡会共享 CPU 和磁盘。P1 用固定两种代表任务同时运行，与单独运行比较慢化比例；若超过 10%，先查 DataLoader/CPU/IO，再冻结 worker 配置。正式成本比较采用预先平衡的混合排程，并记录共跑 run ID。若并行争用仍明显，则同时报告独占基准，避免把外部争用解释为方法优势。

### 8.3 显存、存储和推理

训练至少记录全程 `max_memory_allocated`、`max_memory_reserved`，以及外部 GPU 总显存、CPU RSS。稳定段基准可以在热身后重置峰值，但全程真实峰值另保留，不能丢掉优化器第一次分配造成的峰值。

任务存储与部署存储分别报告：

- LP：头参数与必要元数据是任务增量；部署仍需基础骨干。
- Adapter：Adapter、头及新增 BN buffers 是任务增量；基础骨干另外需要。
- FT/Scratch：任务通常需要完整模型权重。
- 训练续跑状态包括 optimizer、scheduler、RNG 等，不能混同部署文件。

推理采用固定设备、224 输入、batch1 延迟与 batch128 吞吐，明确 FP32 或 BF16（主表统一 BF16）。预热 50 次、测量 200 次，报告延迟中位数和 p95；纯模型计时排除 IO，另给数据到预测的端到端口径。失败 checkpoint 不测正常推理性能。

### 8.4 预算估算与首次执行窗口

P1 对 16 配置各跑 3 个完整 epoch，只使用 train/val，以第 2、3 epoch 的稳定时间并结合启动和存盘成本估算长程任务。每策略采用学习率候选的中间值（FT 指主干值、头乘十），调度为正式 50 epoch 计划的前三轮；profile 权重不进入选参或正式结果。

令 `t_i` 为配置 i 的训练加验证等常规 epoch 成本，额外启动/保存开销独立建模：

\[
\widehat H=\frac{\sum_{i=1}^{16}(3+3\times15+5\times50)t_i
+T_{\mathrm{startup}}+T_{\mathrm{extra\ save}}+T_{\mathrm{eval}}}{3600}.
\]

16 配置合计 **4,768 个计划 epoch**，由短程 48、选参 720、正式 4,000 组成。某项费用已计入 `t_i` 后不得在式外重复加。最后另列有界诊断及 25% 的计划余量，余量不是已发生费用。

情景示例：若加权 epoch 成本为 30 秒，则 4,768 epoch 约 39.7 GPU-hours；若为 120 秒，则约 158.9 GPU-hours。均未含上述额外开销，且不是对这台机器的实测预测。双卡理想完工下界约为总 GPU-hours/2，实际还受排程和共享资源影响。

**首次给 Codex 的执行单元是 P0–P1，建议上限 4 GPU-hours，先返回具体估算。** P2/P3 的总 GPU-hours 与费用不能凭原文分钟数填定；必须在 P1 后根据实测写入预算文件。届时用户可以对这一具体预算审核，ChatGPT 可据此冻结后续执行批次。该科学与成本检查是本方案的阶段安排，本次设计工作已完整给出。

## 九 分阶段执行与验收

| 阶段 | 工作内容 | 计划量 | 验收回交 |
|---|---|---:|---|
| P0 | 环境、数据、权重、模型身份、代码实现和冻结验收 | 不进入长训练 | 环境/数据/权重清单，参数与更新审计，恢复测试 |
| P1 | 短程训练与计时；历史异常的静态排查 | 16×3 epochs | 数值状态、loss/val 曲线、显存、双卡争用、完整预算 |
| P2 | 预先限定的学习率选择 | 48×15 epochs | 全候选表、selected_hparams、配置与预算冻结记录 |
| P3 | 正式五种子矩阵 | 80×50 epochs | 逐 run 原始日志、checkpoint 与状态；集中测试结果 |
| P4 | 统计分析、图表、结论裁定 | 使用 P3 证据 | 完整主表、配对区间、失败统计、资源图与审稿级结论 |
| P5 | 数据量、权重来源和强对照扩展 | 见第十三节 | 独立版本协议，不能混写成主矩阵 |

### 9.1 P0 的必要技术验收

- 模型和数据精确匹配；预训练状态不能默默退化为随机初始化。
- 分类头尺寸、参数数量和训练白名单正确。
- LP 训练 3 次更新后，所有主干参数和 BN buffers 保持不变。
- Adapter 原主干参数/BN buffers 保持不变；新增分支能更新，前部 Adapter 梯度可穿过后续冻结层。
- eval 初始化输出一致；正式训练模式中原 BN 与新增 BN 各自状态正确。
- 输入、标签、loss、logits、梯度与参数有限；标签在 0 至 C−1。
- 累积更新、末尾不完整 batch 和每 epoch scheduler 更新数正确。
- 在固定预生成小批次上，比较连续训练与中断保存后恢复的数步轨迹；若 DataLoader 有随机性，再单独验证 RNG/epoch 恢复机制。
- 输出 schema、状态转换和原子写 checkpoint 正常；失败可保存现场并终止该 run。

这些是直接关系研究有效性的验收，不要求建立与实现机械重复的大型测试体系。

### 9.2 何时自动修复，何时交回审核

依赖缺失、路径错误、日志字段缺失、数据下载失败可在 P0/P1 内修复并重试，保留记录。

发现冻结错误、错误标签、错误归一化、数值实现错误时，先修复并重跑受影响的验收；已经完成的受影响正式结果标为 `invalid_protocol`，不能保留有利部分。

预算超出已冻结上限、需改变训练精度/物理 batch/模型结构、需扩大学习率范围或新增研究任务时，先提交具体差异、代价和建议。技术检查通过后按该阶段清单执行，阶段内无需逐 run 再向用户申请。

## 十 NaN 诊断与历史重建

### 10.1 两条轨道

`legacy_reconstruction` 保留原文可确认的轮数、学习率、AMP、增强与初始化描述，用于检验旧现象；缺失参数在 manifest 中写明。没有原代码时只能称“按文稿重建”。

`canonical_v1` 是本合同正式协议。它已修改验证划分、初始化、BF16、SVHN 翻转与标准化等，因此与旧结果有差异是预期可能性，不能把整体变化当成某个单因素的功效。

历史重建不使用测试成绩来寻找 NaN。重放到相近失败轮次只能帮助定位，不保证第 21 轮准确重复。只跑数百 step 未失败，也不能证明整个 50 epoch 稳定。

### 10.2 记录首次异常

至少区分 forward logits、loss、unscaled gradient、更新后参数及 optimizer state 的首次非有限值，记录：

~~~text
run_id, epoch, micro_step, optimizer_step, sample_ids
loss, logits_min_max, predicted_class_histogram
lr_by_group, parameter_norm, grad_norm_before_clip, grad_norm_after_clip
adapter_output_norm / backbone_output_norm
precision, autocast_state, scaler_scale, skipped_updates
last_finite_checkpoint, optimizer_state, scheduler_state, RNG_state
~~~

常规主实验保留轻量有限值检查；逐层 hooks、异常检测和完整逐步快照只在诊断重放启用，并将额外开销单列。

BF16 主实验中若出现非有限 logits、loss、累积完成后的梯度、参数或 optimizer state，立即将该 run 记为数值失败。非有限梯度必须在 optimizer.step 之前拦截，避免污染原有参数。FP16 诊断的 GradScaler 可以在梯度溢出时跳步；单次跳步不是模型崩溃。若 forward、loss、参数和 optimizer state 仍有限，记录跳步并按既定有限政策继续；连续 20 次 optimizer 更新被跳过则终止为 stalled，保留计数。

FP16 顺序必须是 scale/backward → 累积完成 → unscale → 检查与 clip → scaler.step → scaler.update；不能先 clip 后 unscale，也不能在一个累积窗口中间更改 scale。[S14]

### 10.3 有界干预矩阵

对 ViT-SVHN Adapter 及 ViT-CIFAR Scratch 的旧配置优先检查：

| 编号 | 相对重建基线只改变一项 | 用途 |
|---|---|---|
| N0 | 不改配方，仅加入监测 | 定位首次异常；确认现象是否可重建 |
| N1 | AMP 关闭为 FP32 | 判断精度因素是否参与 |
| N2 | FP16 改 BF16 | 判断 FP16 数值范围是否参与 |
| N3 | 峰值学习率降低十倍 | 判断更新幅度是否参与 |
| N4 | 仅修正 Adapter 输出零初始化 | 判断初始化因素，仅适用于 Adapter |

先完成 N0 和静态检查，再按异常证据决定 N1/N3 的短程重放，N2/N4 作为针对性补充，不一次铺开全网格。初始化对照必须从 epoch0 开始，不能从训练后的状态修改并称为“初始化消融”。

从失败前 checkpoint 继续的局部干预只说明该状态附近的可恢复性；最终选定的稳定配方须从头完整运行目标预算并跨种子检查。多因素交互无法由单因素结果完全解释，需要时再预注册小型二因素对照。

如旧现象无法重建，结论是“在可获得的重建条件下未观察到同类失败；历史原因未唯一识别”。不能编造根因闭环。

## 十一 评价指标与统计分析

### 11.1 模型选择和测试

主指标为 **验证集选出的 checkpoint 在官方 test 上的 Top-1**。不再报告每轮 test 峰值作为主结果。

正式运行过程中保存 best_val 与 last checkpoint。P3 全部运行状态封存、P2 配置与预算未发生未登记变化后，集中评测合法 best_val 模型；last 的测试结果仅为次要稳定性信息。二者都不能反过来选择超参数。

输出逐样本 sample_id、真值标签、预测标签、FP32 logits 或概率及 checkpoint hash。同一 run 产生多个评价记录时，必须保留 evaluation_role（primary_best_val、secondary_last、budget_prefix 或 recovered_failure），不能只保留最有利成绩。

次要指标包括 macro-F1、balanced accuracy、逐类准确率、混淆矩阵与 NLL。SVHN 类分布不均，macro 类指标可以补充 Top-1，但不替代预先确定的主指标。

### 11.2 均值、差值和区间

每个单元报告 5 个原始种子结果、均值、样本标准差、完成次数、失败次数、总计算成本。重复种子体现本划分下训练随机性，不涵盖数据重新划分和预训练不确定性。[S15]

下述正式配对区间与主结论判定，要求所比较双方的五个预注册种子均完整完成、协议有效且评价数值有限。相同骨干、数据、种子的配对差：

\[
\Delta_s=\mathrm{Acc}_{A,s}-\mathrm{Acc}_{FT,s}.
\]

报告差值均值与基于 5 个配对差值的 t 区间：

\[
\bar\Delta\pm 2.776\,s_\Delta/\sqrt5.
\]

单位统一为百分点。五个种子下该区间依赖分布假设，精度仍有限；提供全部差值，不把 p 值当作研究价值的唯一门槛。有缺失或数值失败时，主比较记为不完整，不作正式优越性或非劣性判定。可以附加报告双方共同成功的 m 个种子的条件性描述；如给区间，必须使用真实 m 和 df=m−1，且标为探索性，m<2 不估计训练方差区间。恢复 checkpoint 不混入这五对正常完成的主比较。

若对第 3.2 节两项主比较作显著优越性检验，明确使用双侧配对 t 检验，并对这两项采用 Holm 校正，family alpha=0.05。其余比较标为探索性。五对样本的双侧精确符号检验/Wilcoxon 存在最小 p 值限制，不应临时更换检验去追求显著。

可另做固定模型条件下的按类分层、逐图像配对 bootstrap，用于解释测试样本不确定性；它不是多次训练独立重复，不能把 5×测试图像数作为独立样本数。

### 11.3 实际可接受的精度变化

本方案对预设主比较“ViT IN21K、CIFAR-100、LP 相对 FT”提出实用精度容忍幅度 **0.5 个百分点**，这是项目设计阈值，不是文献公认标准，也不是已证实性能。该正式非劣性判断仍要求双方均 5/5 完整有效；其他组的同类判断标为探索性，不能事后扩大主假设族。

若 LP−FT 的 95% 双侧配对区间下界高于 −0.5，可在上述假设与固定划分下描述为“支持预设容忍范围内的非劣性”；若整个区间在 ±0.5 内，可更谨慎描述差异落在该范围。区间包含 0 不等于等效，区间过宽只能判证据不足。

精度非劣性与节约资源分别判定。可以报告时间下降百分比、显存下降百分比及其逐种子结果，但不能用很少参数自动替代实测资源证据。

### 11.4 失败不能隐藏

状态至少包括：

~~~text
planned / running / completed / blocked / cancelled
numerical_failure / stalled / oom
infrastructure_interrupted / invalid_protocol
~~~

主表保留计划 n=5 的分母。成功运行条件下的均值必须标注“成功 k/5”；不能静默丢弃失败种子，也不能把 NaN 当作 0% 准确率混入普通均值。

若中途失败前保存了合法 best_val checkpoint，按预定规则可以单列其恢复后性能，同时保留训练失败状态。这不是“正常完成”，也不能一概断言该 checkpoint 完全不可用。

不对非有限 logits 继续 argmax 并把单类坍缩后的 6.70% 写成正常 Final accuracy。历史第 21 轮 NaN 不能直接称为“梯度消失”。

0/5 失败仅表示本轮没有观察到失败；其可靠性证据仍有限，不能推断一般失败率为零。

## 十二 图表与解释证据

### 12.1 必交图表

| 图表 | 内容 | 支持的判断 |
|---|---|---|
| 主结果表 | 四个 dataset/backbone 组，四策略，5 seeds、均值±SD、完成率 | 性能差异与稳定性 |
| 学习曲线 | train loss、val loss、val Top-1，均值及种子轨迹 | 收敛、过拟合线索与异常 |
| 配对差值图 | 两个预设主比较及探索性比较 | 差异大小与不确定性 |
| 精度与训练成本图 | accuracy 对 GPU-hours/fit time | 时间代价下的效果 |
| 精度与显存图 | accuracy 对 peak allocated/reserved | 实际训练内存折中 |
| 参数与存储表 | base/added/trainable/total、任务增量、完整部署文件 | 参数效率与存储范围 |
| 混淆矩阵 | 固定模型的逐类错误 | 错误类型，尤其 SVHN |
| NaN 轨迹图 | 首次异常附近数值与单因素干预 | 对已定位触发因素的有限解释 |

训练曲线使用验证集，不能逐 epoch 画 test 曲线。论文图保留数据表和可重跑绘图脚本，导出矢量 PDF/SVG 与高分辨率 PNG。

### 12.2 共同时间预算切片

为避免额外铺开一套等时训练，先使用固定 50 epoch 配方的 anytime 前缀。

在 P3 前按 P1/P2 计时冻结每个 dataset/backbone 组的共同上限 B：取四策略预计 50 epoch fit time 的最小值，并设置 0.25B、0.5B、B 三个切片。B 只能由计时决定，不看测试精度。

每个切片使用该时刻之前已完成验证、按同一规则选定并保存的 checkpoint；其选择和必要存盘成本也须在时间上限内。若预算内尚无已完成验证的模型，记为不可得，不用下一轮 checkpoint 顶替。训练结束早于切片时记录实际停止时间，不虚构多余计算。

保存每个切片对应的 checkpoint 身份，在最终集中测试阶段评测。必须称为“本固定训练配方在共同资源上限下的可达结果”，不能声称每个时长都独立重新调参到了最优。

### 12.3 热力图

主可视化优先使用两类架构均可计算的遮挡敏感性图：在同一 224 输入上，按固定 16×16 窗口、stride16 用归一化零值遮挡，计算固定目标类别 logit 的变化。使用同一图像、同一目标类别、相同遮挡和颜色规则比较四策略。

这只是遮挡干预下的局部敏感性，可能引入分布外图像，不能直接证明真实因果关注点。Grad-CAM 或 ViT attention 可以作为附图，不把二者的原始数值当成统一标尺。

P0 预先固定展示池，例如每个数据集 20 个验证样本，按类别/索引规则抽取；正式模型统一使用 seed42，禁止事后挑最好种子。测试解锁后可以按预先定义的共同正确、共同错误、方法分歧分组展示，给出候选池大小、抽取规则与全部索引。

CIFAR-100 和 cropped SVHN 没有直接用于此实验的统一目标区域标注；漂亮热力图不足以证明模型学到了“正确关注点”。若要提出定位结论，需要独立标注与定量定位指标，另立实验。

## 十三 扩展实验的优先顺序

所有扩展保留主矩阵身份，新增单独配置、成本和结论范围。以下正式 run 数不包含新增选参。

| 优先级 | 扩展 | 新增正式运行 | 解决的问题 |
|---|---|---:|---|
| A1 | CIFAR-100 仅用本轮训练部分的 10% | 2 骨干×4 策略×5 seeds=40 | 下游标注样本减少后，方法排序是否改变 |
| A2 | ViT 同架构 IN1K 权重控制 | 2 数据集×3 迁移策略×5 seeds=30 | 同一架构下，不同预训练来源是否改变迁移结果 |
| B | 已有强 Adapter 或简单 PEFT 基线 | 先限定 ViT、CIFAR-100，再写具体矩阵 | 避免把自定义较弱实现等同整个 Adapter 家族 |
| B | R50 Adapter 容量对照 | 先限定 SVHN，5 seeds=5 | 检查原 30% 训练占比是否必要 |
| C | Scratch 更长预算 | 对指定模型成组扩展 | 区分有限预算收敛慢与不可训练 |
| C | 第三个分类数据集 | 冻结官方数据与拆分后制定 | 外部泛化证据；无需立即扩检测/分割 |

**10% 数据扩展：**从固定的 45,000 训练样本中每类取 45，共 4,500。不能每类取 50 后仍称 10%。验证集 5,000 与官方测试保持不变。首次以主协议选定超参数固定迁移，解释为同配方下的数据预算效应；若之后重新调参，另记为数据预算下的优化配方结果。固定子集不覆盖抽样不确定性，可在后续对照中增加独立子集。

固定 50 epochs 时，减少训练样本也会减少 optimizer updates 与样本处理总量，因此 10% 扩展反映固定 epoch 配方的少样本表现，不能拆解为数据多样性与优化步数两个独立因果效应。该扩展仍保留 5,000 个验证标签，开发阶段共使用 4,500+5,000=9,500 个监督标签；不能称为总共只使用 4,500 个标注或总标注成本减少 90%。

**预训练来源扩展：**使用 `vit_base_patch16_224.augreg_in1k`，它的来源是 IN1K。不能用 `augreg_in21k_ft_in1k` 冒充无 IN21K 的对照。三种迁移方法需按相同有限规则为新权重独立选参（另有 18 次 15 epoch 试验），Scratch 在相同架构和输入设置下复用，不重复计算。[S16]

该对照仍不能将全部差异识别为纯预训练数据规模效应：不同 checkpoint 的配方、来源细节仍可能不同。若要估计纯规模因果效应，需要自行控制预训练实验，当前不开展。

**强对照：**优先核对官方 AdaptFormer，或根据论文定位选择 LN-Tune。它们单独列为第五策略，不替换当前主实验。[S2,S10]

**容量对照：**R50 每个 Adapter 用固定瓶颈宽度 32 时，新增参数预计为 997,888；SVHN 训练参数约 1,018,378，含新增模块后的占比约 4.15%，接近原 ViT Adapter 的约 4%。这有助于解释容量差异，但并不等价于相同 FLOPs 或显存。

原稿附后的建议提出增加检测、分割和更多数据集。本合同将“补齐可信分类证据”排在前面；跨任务扩展需要新数据、头、损失和指标，会改变研究范围，应由用户结合首轮结果决定。

## 十四 代码结构与证据交付

建议项目目录为 `/root/autodl-tmp/tl_cls_benchmark`。这只是新项目建议路径，Codex 应检查已有内容，避免覆盖其他项目；版本管理若已有仓库则沿用独立分支，未提供仓库时可先建立本地项目，不自动推送或公开数据。

| 路径或文件 | 内容 |
|---|---|
| `contracts/protocol.md` | 本合同及修订历史 |
| `configs/protocol.json` | 共同设置、候选范围、冻结字段 |
| `manifests/run_matrix.json` | 完整计划与状态，不凭成功日志推算分母 |
| `manifests/datasets.json` | 来源、hash、标签、拆分索引及训练子集 |
| `manifests/weights.json` | exact ID、revision、SHA、预训练来源、构造参数 |
| `manifests/selected_hparams.json` | 全候选及选择依据 |
| `records/preflight.json` | 实测环境与配额 |
| `records/implementation_checks.json` | 参数、BN、梯度、恢复验收 |
| `records/budget_plan.json` | P1 计时、P2/P3 GPU-hours、费用和存储估算 |
| `src/` | 数据、模型、策略、训练、评价、成本记录实现 |
| `runs/<run_id>/config.json` | 完整解析配置与协议 hash |
| `runs/<run_id>/epochs.jsonl` | 每 epoch 原始 loss、val、学习率、时间、状态 |
| `runs/<run_id>/summary.json` | 参数、资源、状态、checkpoint 与评价入口 |
| `runs/<run_id>/predictions.*` | 集中测试后的逐样本输出 |
| `reports/` | 阶段回执、全结果表、图表与科学结论 |

保持两个可复现入口：按 run ID 训练，以及按已冻结 evaluation manifest 评价。训练入口默认无测试集访问路径；评价入口不能改动模型或调参。

checkpoint 采用临时文件后原子替换；至少保留 best_val、last 和已冻结预算切片需要的权重。自包含模型 checkpoint 使用完整 model.state_dict()，包括 BN running_mean、running_var、num_batches_tracked。若保存任务增量，则包含 head、全部 Adapter 参数及其 buffers，通过基础权重 hash 绑定主干；不能只按 requires_grad 筛选后漏掉 BN 统计。

续跑状态包含 optimizer、scheduler、epoch/update、RNG 与数据顺序。构造并加载模型、优化器等以后、取下一批之前恢复 RNG；按策略重新应用 requires_grad 和 train/eval，因为 state_dict 不自动恢复这些模式。先验收加载前后 eval logits 一致，再验证继续更新的轨迹。常规续跑从最后完整保存的 epoch 边界开始，epoch 中断则按固定样本/增强顺序重放未完成 epoch；重算 GPU 时间保留在成本中。诊断性中途快照另记录梯度累积边界，不假设能无损接上半个累积窗口。

恢复前后 fit 计时按实际执行段累计，不清零；未运行的停机等待单列 calendar elapsed，不计为 GPU 工作时间。不要每个 epoch 无限复制完整模型。

每个文件须带 code commit、protocol hash、split hash、weight hash 与实际启动参数。依赖清单、包版本、CUDA/cuDNN 信息随回执保存。未经运行的字段使用 null/planned，禁止填入原论文数值冒充新结果。

### 14.1 Codex 第一轮回执

必须回交：

1. P0/P1 状态、代码位置与 commit、当前完整配置。
2. 环境实测与用户清单的差异；单卡可用显存、容器配额、真实磁盘预算。
3. 数据与权重身份、split hash、参数整数与冻结/更新验收。
4. 16 个短程配置的完成状态、3 epoch 日志、显存、每 epoch 时间。
5. 原稿 NaN 的静态检查结论；有无旧代码，重建与精确复现的区分。
6. 48 次选参、80 次正式训练的 GPU-hours 估算、两卡排程、25% 余量与需要的磁盘空间。
7. 尚未解决的具体问题及最小修复建议，不以“整体正常”代替证据。

### 14.2 正式结果回执

提交完整结果表、逐种子记录、失败列表、所有计划 run 的状态、逐样本预测、checkpoint hash、训练曲线和成本分解。ChatGPT 按第十一节分析，用户审核结论与后续方向。

验收不要求超过 88.65%、96.25% 等旧数值。只要协议有效，方法效果较弱、旧现象未复现或假设被推翻，均属于有效实验结果。

## 十五 可直接下发给 Codex 的首轮任务

~~~text
请执行 TL-CLS-20261020-v1.0 的 P0–P1。

目标：建立 SVHN/CIFAR-100、ResNet-50/ViT-B16、Scratch/Full FT/严格 LP/
canonical Adapter 的可信实验管线，先完成环境、数据、权重、参数冻结和短程计时。

读取随附实验设计合同、protocol.json、run_matrix.json 和结果字段模板。
创建独立项目，复用已工作的 torch2.5.1+cu124 与 torchvision0.20.1+cu124；
不得静默更换指定预训练权重。先核验输入标准化、224尺寸、SVHN标签与取消翻转。

先做参数白名单、BN状态、初始化恒等、Adapter梯度连通、累积末批次及恢复验收。
通过后，按清单对16种配置分别以profile seed314159运行3个完整epoch，
仅用train/val，实测显存、时间、双卡争用和checkpoint成本。
首批建议上限4 GPU-hours；如无法在该窗口内完成，保存状态并报告具体剩余量。

不使用测试集选模，不用旧论文成绩作为新结果，不把错误Partial FT算作LP，
不把NaN后的argmax准确率作为正常指标。

回交P0/P1报告和原始证据，给出48次选参+80次正式运行的实测外推预算，
供实验负责人核查和用户审核长期成本。暂不把未知总预算填成已确认。
阶段内常规技术修复自行完成并记录；会改变科学口径的修改先给差异说明。
~~~

## 十六 本协议与旧配方的差异登记

| 变量 | 原稿 | 本合同 |
|---|---|---|
| 重复次数 | 1 seed | 正式 5 seeds，独立选参 seed |
| 训练/验证/测试 | 未明确独立验证，逐轮测试峰值 | 固定 train/val/test；val 选模；集中 test |
| 主矩阵 | 11 个成功且策略正确单元 | 16 配置×5 seeds，失败仍保留分母 |
| Epoch | 100/30/50/50 | 主矩阵全部 50 |
| Batch | 随架构和策略变化 | micro32×4，有效128 |
| 学习率 | 每策略单值 | 每单元3候选，FT head倍率固定 |
| AMP | 未明确细节 | 主矩阵 BF16；旧 FP16 单独诊断 |
| 标准化 | 目标数据集统计，具体值缺失 | 指定权重推荐值；Scratch同骨干一致 |
| SVHN翻转 | 有 | 无 |
| Adapter初始化 | 细节不足 | 显式零输出分支并验收 |
| 耗时 | 总分钟，边界不足 | 训练/验证/存盘/总适配/调参/失败成本拆分 |

新结果不得与旧表混成同协议平均值。所有实质变化通过本表公开，保留旧观察的历史身份。

## 十七 核验的一手资料

访问日期统一为 2026-10-20。以下是与本方案设计直接相关的资料，不是为了满足固定篇数的论文参考文献清单。

- **[S1]** Touvron 等，Training data-efficient image transformers & distillation through attention，ICML 2021。[论文](https://proceedings.mlr.press/v139/touvron21a.html)；[作者代码](https://github.com/facebookresearch/deit)。用于约束“无预训练 ViT 不可训练”的过度结论。
- **[S2]** Basu 等，Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning，AAAI 2024。[论文原页](https://ojs.aaai.org/index.php/AAAI/article/view/28978)。用于公平比较与简单强基线。
- **[S3]** Mercea 等，Time-, Memory- and Parameter-Efficient Visual Adaptation，CVPR 2024。[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Mercea_Time-_Memory-_and_Parameter-Efficient_Visual_Adaptation_CVPR_2024_paper.pdf)。用于区分参数、时间和内存效率。
- **[S4]** Yi Xin 等，Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey and Benchmark。[arXiv 2402.02242](https://arxiv.org/abs/2402.02242)，页面显示 2024 首发、2025-12-09 修订 v7，Comments 为 Submitted to IJCV；[作者资源库](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning)。不能直接沿用原文“Han 等、TPAMI 2024”的条目信息。
- **[S5]** [CIFAR 官方数据页面](https://www.cs.toronto.edu/~kriz/cifar.html)。用于类别、样本数和文件校验。
- **[S6]** [TorchVision 0.20 SVHN 源码](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/svhn.html)。用于文件 hash、标签映射和 SciPy 依赖。
- **[S7]** [timm ResNet50 A1 模型卡](https://huggingface.co/timm/resnet50.a1_in1k)；[配置](https://huggingface.co/timm/resnet50.a1_in1k/blob/main/config.json)。用于模型来源、标准化与默认测试尺寸。
- **[S8]** [timm ViT AugReg IN21K 模型卡](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k)；[配置](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/blob/main/config.json)。用于模型来源、分类头与标准化。
- **[S9]** [PyTorch v2.5.1 Autograd 官方说明源码](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/autograd.rst)。用于 requires_grad、grad mode 和 eval 的区分。
- **[S10]** AdaptFormer：[Adapter 实现](https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py)；[图像训练入口](https://github.com/ShoufaChen/AdaptFormer/blob/main/main_image.py)。用于初始化与模块定义，执行时另固定 commit。
- **[S11]** [PyTorch v2.5.1 随机性说明源码](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)。用于复现范围与随机状态控制。
- **[S12]** [PyTorch 官方历史版本与安装配对](https://pytorch.org/get-started/previous-versions/)。用于 torch2.5.1、torchvision0.20.1、cu124 配对。
- **[S13]** [timm 1.0.12 发布页](https://pypi.org/project/timm/1.0.12/)。实际兼容性以 P0/P1 验收为准。
- **[S14]** [PyTorch v2.5.1 AMP examples 官方说明源码](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)。用于 unscale、clip、累积和溢出跳步。
- **[S15]** Bouthillier 等，Accounting for Variance in Machine Learning Benchmarks，MLSys 2021。[论文原页](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)。用于训练、采样与超参数选择的方差来源。
- **[S16]** [timm ViT AugReg IN1K 模型卡](https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k)。用于同架构预训练来源控制。

## 十八 尚需运行证据才能填定的字段

下列信息不能现在虚构：服务器实际连通和配额、原始实验代码是否可得、指定权重的本地 SHA/revision、依赖冻结收据、每个配置真实显存和吞吐、选定学习率、长期训练 GPU-hours 与费用、所有正式测试结果。

这些字段已在配套 JSON 中保留为 null 或阶段性待填项。它们是 P0–P3 的明确产出，不影响当前实验设计与首轮任务的完整下发。
