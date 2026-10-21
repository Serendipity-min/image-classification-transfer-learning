# P2 阶段启动审查与 Codex 执行补充 v1

日期：2026-10-21  
实验协议：`TL-CLS-20261020-v1.0`  
审核仓库：`Serendipity-min/image-classification-transfer-learning`  
审核提交：`b485177a393b8532e07d1504a64629ad54043d2b`

**科研审核结论：接受 P0/P1 工程验收；建议进入 P2 单阶段，启动前完成本文件列出的入口与恢复检查；P3 保持待定。**

本文件是审核意见与建议执行条款，供 PI 审定后交给 Codex。它不代填 PI 身份，不代表已经修改阶段授权，也不代表 P2/P3 已经执行。本次审核没有连接训练服务器、启动训练、修改远程仓库或删除 checkpoint。

## 1. 阶段决定

| 阶段 | 审核意见 | 下一步 |
|---|---|---|
| P0 | 接受现有实现验收证据 | 保留原始记录；新执行代码按变更范围重新绑定验收 |
| P1 | 接受 16 组、48 个完整 epoch 的短程工程证据 | 无需重跑全部 P1；不得当作正式五种子结论 |
| P2 | 有条件进入 | 完成第 4 节启动检查，在独立资源上限内执行 48 个候选 |
| P3 | 暂缓 | 等待 P2 选参、长程预算、时间切片和存储策略冻结后审核 |
| 集中测试 | 本阶段关闭 | 保留 P3 封存后统一测试的既定顺序 |

P2/P3 入口尚未实现，属于下一阶段工作；恢复和测试路径中的缺陷没有发生在现有 P1 的正常运行里。因此不应据此宣布 P1 无效，也不应直接将现有 P1 入口改成无约束长程执行。

## 2. 审核依据与证据范围

通过 GitHub 读取固定提交；仓库元数据确认可见性为 PRIVATE，审核时 main 指向上述提交。读取的 133 份文本文件均按原始字节重算 Git blob SHA，与远程一致。其中 123 份属于服务器 evidence manifest，逐份 SHA-256 和长度也一致；该 manifest 其余 87 个条目未在本次逐文件下载校验。

全部 16 组 P1 的 config、epochs、summary、result_record、active_costs 均已读取。以下结论来自日志独立复算和源码审查，不只依赖 PI 摘要：

- 16 个配置各完成 3 轮，共 48 轮；累计处理训练样本 2,662,344 次，optimizer 更新 20,832 次。
- CIFAR-100 每轮训练 45,000、验证 5,000，352 次更新，末累积窗口 72 样本。
- SVHN 每轮训练 65,931、验证 7,326，516 次更新，末累积窗口 11 样本。
- best-val 及并列选择较早 epoch 的规则可以从逐轮日志复算；CSV 与 summary 一致。
- 同数据集、同骨干的四策略初始分类头 hash 一致；相同数据集和 epoch 的 sampler hash 与末窗口 sample IDs 一致。
- 每种策略在两张物理 GPU 上各执行 2 组；全部 P1 都是 resume=false。
- 日志中 loss、准确率有限，未记录 OOM/非有限值；训练入口只有 train/validation，全部 P1 测试成绩为空。
- LP 的冻结、Adapter 的位置/初始化/梯度路径、新旧 BN 模式、真实尾批归一化、独立增强随机流符合合同。
- 父台账记录 P0/P1 共 1.5233777703 GPU-hours，低于 4 GPU-hours。
- 实际硬件与原清单一致：双 RTX 3090、各 24 GiB，PCIe 1E/3D，16 vCPU、64 GiB 内存配额。
- 六组运行源码身份为 74001fc，十组为 94b02df。对两个历史 tree 和当前 tree 的逐文件比较确认，训练、模型、数据及协议源码一致；差异为 P0/P1 调度器。归档 commit 不应回填到历史 run 身份。

本次没有独立加载服务器上的 40 个 checkpoint 二进制；其存在、大小和 SHA 依据仓库中的服务器回执。这不影响逐轮日志与实现的审查，但不能表述为本次已在服务器重新运行全部验收。

原始执行合同 SHA-256 为：
`e19e65e8ea85d8a6c29266b61ad47d893cbf9b608ce3423eb0b3cbbc0c597298`。
仓库版本与此前交付的合同字节一致。

## 3. P1 结果的解释边界

P1 只有一个 profile seed，且三轮均处于五轮 warmup 内。它支持工程可运行性、短程吞吐和资源估算；不足以支持最终方法排序、五种子统计推断、50 轮稳定性或历史 NaN 的单因素归因。

例如，CIFAR-100/ViT 的 Full FT 三轮最佳验证准确率为 91.5%，SVHN/R50 的 Full FT 为 54.245%。这些数字应完整保留，但不据此修改学习率网格、删除某种方法或宣布最终赢家。

CIFAR-100 有 14 组跨划分精确像素重复，具体为 4 组 train↔validation、10 组 train↔test；SVHN 的一组重复仅发生在训练内部。这是原数据的重复样本问题，不等于 P1 用测试成绩选参。按已冻结合同继续 audit-only，不静默删除样本或重划分；论文应披露这一局限，不能将索引不交叉表述为像素完全独立。

跨 R50 与 ViT 的比较继续限定为两个具体管线；架构、规模、预训练来源与标准化共同变化，不能单独识别架构或预训练数据量的因果效应。两种 Adapter 也不具有相同结构或容量。

## 4. P2 GPU 启动前必须完成的工作

### 4.1 实现明确的 P2 入口、调度和报告

现有 `src/train.py` 第 53 行只接受 P1，第 67–68 行写死 phase=P1、planned_epochs=3。`src/execute_p0_p1.py` 也只调度 P1，CAP 为 4 GPU-hours。现有预算 JSON 不能代替可执行的预算控制。

Codex 应新增 P2 阶段分支或独立入口，读取原矩阵中的 48 条 P2 run；phase、epochs 和报告字段从经过验证的配置取得。新增独立 P2 台账与阶段决定文件，保留 P0/P1 原台账。P3 及测试保持关闭。

不得只删除 phase 断言、把旧 CAP 改大，或将 P1 报告标签改成 P2。启动前检查 48 个 ID 唯一、每配置三档 LR、seed=123、epochs=15、scheduler horizon=50、warmup=5。

### 4.2 实现确定性的 P2 选参器

选参器读取完整日志，验证候选配置身份、15 个完整 epoch、有限值和合法完成状态，再按第 13、14、15 轮验证 Top-1 的算术平均值选 LR；并列选择数值较小的 LR。

同一配置的验证样本数恒定，可用三轮 correct 之和进行精确比较，避免显示小数的四舍五入制造并列。不得用 15 轮最高值、test 成绩、最后一轮单点或训练 loss 替代选择规则。

数值失败候选保留记录但不具备正常选参资格。基础设施中断应在预算内按原科学配置恢复；若预算耗尽且候选未完成，配置标为未完成，不能把它伪装为数值失败后静默选择幸存者。三个候选均失败则该配置 blocked，回报具体原因，不临时扩展 LR 网格。

生成根目录 `selected_hparams.json`，其结构兼容原 P3 的 selected_lr_ref。每条选择至少记录 configuration_id、三条候选身份及状态、13–15 轮值/均值、选中 run、选中 LR、tie-break、协议/数据/权重/科学配置 hash 和源日志 hash。文件生成不触发 P3。

### 4.3 修复恢复的多文件提交一致性

目前训练顺序为覆盖 best、覆盖 last、写 active_costs、追加 epoch 日志。在这些写入之间被终止，可能出现 last 已前进一轮、epoch 日志缺行或旧 sidecar 回滚成本。

应以一个原子提交清单绑定 checkpoint 代次、completed_epoch、best 身份、完整 epoch row 和累计成本。恢复以最近完整提交为准，日志补写必须幂等。未提交代次不能成为有效恢复边界；sidecar 只能在代次/hash 匹配且时间单调时合并。

为科学配置生成独立稳定 hash，纳入 LR、候选索引、训练轮数、batch/累积、精度、优化/调度、数据/权重/依赖与实现身份。将设备、启动时间、剩余预算等运行元数据分开。恢复时拒绝科学配置不一致；不能让日志显示新 LR，而 optimizer 实际恢复旧 LR。

保存和恢复完整 model state、BN buffers、optimizer、scheduler、RNG；构造后重设 requires_grad/train/eval 模式，最后恢复 RNG，沿用原合同。

### 4.4 补齐父子进程预算与异常退出控制

SIGTERM/SIGKILL 不保证子进程 finally 执行。父账本必须按 run+attempt 保存实际占用时间，恢复时与子记录对账，避免遗漏被中断半轮成本或重复累计。

双卡采用一个共享台账，累计各 GPU 实际占用；两张卡并行一小时约记两 GPU-hours。父进程异常退出时清理其子进程组；统一退出期限和宽限时间，不能逐进程等待导致越过总上限。按退出实测设置足够的总 GPU 秒余量，初始至少预留 60 GPU 秒，并将退出/drain 时间计入台账。

不得因预算不足缩短候选训练、跳过验证或将半轮标成完整完成。新增任务前应能覆盖到下一合法保存边界和退出余量。

### 4.5 验收新路径，不重跑全部 P1

先完成 CPU 级检查：48-run dry-run、选参规则/并列/失败样例、各写入边界故障注入、科学配置不匹配拒绝、父子进程终止与台账对账。相关测试只覆盖新增风险，不写镜像实现的空洞测试。

新代码提交必须关联真实验收。可运行现有短 P0 实现检查并另存新回执，或用严格的代码范围 hash 说明未改的数学管线与已验收身份一致；不得直接改旧回执的 code_commit 来伪造验收。任何必要 GPU 验收均计入 P2 的 30 GPU-hours。

同时纠正资源报告字段：当前结束时 cpu_rss 不应填成 peak_cpu_rss。应准确标注终值/主进程采样峰值，或测量包含 workers 的进程树/cgroup 峰值并注明方法。该口径修正不要求重做 P1。

## 5. P2 冻结实验设置

| 项目 | 设置 |
|---|---|
| 配置 | 2 数据集 × 2 骨干 × 4 方法 = 16 |
| 候选 | 每配置 3 个预注册 LR，共 48 个 run |
| 随机种子 | 123 |
| 训练长度 | 每候选 15 个完整 epoch，共计划 720 epoch |
| 调度 | 50-epoch horizon 的前 15 轮；5 轮 warmup |
| 物理 batch / 累积 | 32 / 4，有效 batch 128；尾窗口按真实样本数 |
| 精度 | BF16 autocast，FP32 参数/loss；TF32 关闭 |
| GPU | 每 run 一张卡，最多两任务并行；记录并均衡设备分配 |
| 初始化 | 各候选全新初始化；迁移方法从冻结基础权重开始，Scratch 随机初始化；不续训 P1 |
| 验证选参 | epochs 13–15 验证 Top-1 均值最大；并列小 LR |
| 测试 | P2 不评价 test，不输出 test 预测 |
| 数据/增强/Adapter | 全部沿用原协议与 hash，不按 P1 分数临时改变 |

| 方法 | 三档 LR | 分类头 LR |
|---|---|---|
| Scratch | 3e-4、1e-3、3e-3 | 同主干 |
| Full FT | 1e-5、3e-5、1e-4 | 主干 LR × 10 |
| Linear Probe | 1e-3、3e-3、1e-2 | 唯一训练模块 |
| Adapter | 1e-4、3e-4、1e-3 | 同 Adapter |

候选出现非有限值，按原协议停止该候选并保留现场。不得偷偷降 LR、改精度、改 batch 后沿用同一科学身份。技术重试必须登记；不得为获得更好成绩重复运行同一候选。

## 6. P2 资源上限和停止条件

### 6.1 GPU-hours：建议总硬上限 30

逐配置使用 P1 第 2–3 轮的 train+validation+checkpoint 均值 t，固定开销 a 另含启动、终结和首轮超额。独立复算：

P2 = Σ配置 [3 × (15t + a)] / 3600 = **21.5040478821 GPU-hours**。

原 26.8800598526 = 21.5040478821 × 1.25，属于单卡独占基线加余量。当前报告把代表并发慢化乘到了 wall time，没有乘到 GPU-hours；因此不能同时声称 26.88 已覆盖并发且仍有完整 25% 余量。

采用两个短测中的较大慢化因子 1.0787750477 作为规划情景：

| 项目 | GPU-hours |
|---|---:|
| 单卡独占基线 | 21.5040 |
| 乘代表并发情景 | 23.1980 |
| 再加 25% 余量 | 28.9975 |
| 建议 P2 全部活动硬封顶 | **30.0000** |

30 包含该阶段 GPU 验收、所有候选、失败、恢复、技术重跑、必要诊断和退出开销；不额外叠加原提案中的 2 小时诊断，不挪用 P0/P1 未用完的预算。预算余量是规划空间，不能当作已经消耗的算力写进实际台账。

按当前排程，两卡 P2 约 11.6238 小时；同一情景加 25% 约 14.5298 小时。它是资源可用且无额外中断的估算，不是完工保证。原 94.9204 小时对应 P2+P3 训练，不是 P2。

代表并发微基准每卡仅 6 次有效更新、768 样本，约 1.2–1.8 秒，不覆盖整轮验证/存盘。用首批已登记 P2 候选的第 2–3 个完整 epoch 重新校准成本，无需另增加 pilot。超过对应 P1 基线 10% 调查线时暂停发新任务，核查共享 CPU、数据加载、IO 和其他负载；预测总成本超 30 时同样暂停，不静默扩大额度或改变矩阵。

### 6.2 存储：本阶段可以完整保留证据

| 项目 | GiB |
|---|---:|
| 当前项目，已含数据、权重及 P1 | 11.3702 |
| P2 新增 best、resumable last、LP/Adapter delta | 30.9313 |
| 独立环境新增非链接文件 | 0.1141 |
| 合计 | 42.4157 |
| 再加 25% | **53.0196** |
| 建议确认的项目及环境可用分配额度 | **至少 60** |

保留 P1 全部 40 份 checkpoint；P2 用新 run 目录，全候选保留原始日志、best、last，LP/Adapter 保留含 buffers 的任务增量。按当前大小估算，P2 新增 120 份常规 checkpoint。数据和基础权重共享，不按 run 重复复制。不为本轮删除 optimizer 或未选中候选。

原 117.32 GiB 是 P1/P2/P3 全保留的整个项目加余量；180.45 GiB 还包括每个 P3 run 额外三份完整切片。两者属于 P3 的存储决策，不是 P2 当前的容量门槛。

Codex 应读取实际项目/租户配额或明确的分配记录；目前 quota=null，不能把共享盘剩余约 15 TB 写成项目额度。建议 P2 内部预警 54 GiB、保护线 60 GiB，仍不突破原 100 GiB 规划线；如果实际分配更小，取更严格限制。预留至少 5 GiB 可写空间用于双任务临时原子文件及退出，额度与物理剩余空间都要检查。入口无法确认资源保障时回报这一具体阻塞，不自动删除证据。

学校服务器无计价，金额项不适用；仍按 GPU-hours、占用时间和存储审核。

## 7. P2 完成后的交付

Codex 提交一次完整阶段回执及固定 Git commit，至少包含：

1. 新执行提交、协议/数据/权重/依赖锁/科学配置 hash，新增路径验收和恢复故障注入记录。
2. 48 条候选的 planned→terminal 状态、每轮日志、验证曲线、三档 LR 比较表，失败及技术重试清单。
3. 可重建的 selected_hparams.json，含全部候选成绩和精确选择依据；不得只提交赢家。
4. 独立 P2 GPU 台账，实际总成本、首批长窗口并发校准、各 run 与 attempt 占用和退出成本。
5. 服务器 checkpoint 清单和 SHA、存储实测，报告/源码/配置/日志推送到现有私有仓库；大权重继续留服务器。
6. P3 的 80 条解析后配置草案、真实 P2 成本外推、时间切片方案、存储/评价数量及预算提案。保持 P3 未启动。

基础设施中断未完成的候选、数值失败、全部失败的配置都要显式呈现。完整回执不等于 48 个候选必须全部成功；任何未完成状态也不等于可以自动选择幸存候选并升级 P3。

## 8. P3 再审核的必要条件

- 16 个配置的 LR 和选择证据冻结，P3 解析所选 LR；原 80 个正式 run 使用种子 17、42、2026、3407、65537，从原始初始化重新训练 50 轮。
- 依据原合同及 P2 时序证据冻结四个 dataset/backbone 组的共同预算 B，以及 0.25B、0.5B、B。说明计算口径，不能为观察到的精度改变预算选择规则。
- 每个预算点只使用在预算内已完成验证且存盘提交的 checkpoint；保存不可变身份和提交累计时间，防止后续 best 覆盖。无合格 checkpoint 时报告不可用，不外推。
- 完成 P3 seal 生成器和冻结 evaluation manifest；80 条计划运行按规则封存终态后才能集中测试。失败仍保留正式分母，不以幸存子集冒充完整五种子分析。
- 修复 evaluate.py 当前仅用 run_id+evaluation_role 命名的覆盖问题：三个 budget_prefix 会重名。使用唯一 evaluation_id，并按 run_id+checkpoint_sha 去重复用预测，多个预算点分别引用。
- 在构造 TestData 前校验 manifest、checkpoint 内 run/phase/config、选择身份、状态、角色、预算提交时间和 hash 的一致性，拒绝错 run、错阶段、未冻结选择及无效协议产物。
- 用 P2 实测和新的保留政策重算 P3 GPU-hours 与存储，再由 PI 决定。原 179.59 GPU-hours 未显式包含训练并发慢化，也未包含尚未冻结的切片工作，不能直接作为 P3 全包额度。
- 主比较、五种子配对分析、置信区间、Holm 校正和非劣效界限沿用原合同；不依据 P2 胜负临时更换主要假设。

## 9. 可转交 Codex 的阶段指令

以下为供 PI 采用的建议文本，实际授权以 PI 下发为准：

> 接受 b485177 的 P0/P1 工程验收，下一阶段范围为 P2。先完成 P2 入口/调度/报告、确定性选参、恢复事务与配置身份、共享预算及异常退出检查，并将新代码与验收绑定。
>
> 启动检查通过后执行原矩阵 48 个 P2 候选：seed=123，每候选 15 epochs，50-epoch scheduler horizon、5-epoch warmup；按 epochs 13–15 验证 Top-1 均值选 LR，并列小 LR。全新初始化，不续训 P1，不访问测试集。
>
> P2 独立总硬封顶为 30 GPU-hours，验收、失败、恢复、技术重跑和诊断全部包含。确认至少 60 GiB 项目及环境分配或等效资源保障，保留所有 P1/P2 证据；设置存储和退出保护。使用首批已登记候选的完整 epochs 2–3 校准成本，超过 10% 慢化调查线、预测超预算或资源不足时暂停发新任务并报告。
>
> 交付全部候选、选参文件、GPU/存储台账、checkpoint 清单、新 commit 和 P3 提案。P2 完成后停止，P3 与集中测试保持关闭，等待下一次阶段审核。

## 10. 固定提交的主要证据链接

- [PI 审核入口](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/docs/PI_审核与阶段分析.md)
- [验收审计](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/records/acceptance_audit.json)
- [P1 完整配置表](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/reports/P1_acceptance_table.csv)
- [实现验收](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/records/implementation_checks.json)
- [逐配置预算与排程](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/records/budget_plan.json)
- [附加预算和存储](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/reports/acceptance_addendum.json)
- [训练入口](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/src/train.py)
- [训练与恢复引擎](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/src/tlcls/engine.py)
- [父进程调度](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/src/execute_p0_p1.py)
- [集中评价入口](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/src/evaluate.py)
- [原协议](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/configs/protocol.json)
- [原运行矩阵](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/b485177a393b8532e07d1504a64629ad54043d2b/configs/run_matrix.json)

