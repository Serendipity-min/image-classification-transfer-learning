# TL-CLS｜P4-B2 Flowers/ResNet-50 Full FT 学习率边界补证与 P4-D 统一论文重写执行合同

- **版本：**`TL-CLS-P4B2-20261028-v1.0`
- **依据：**GitHub `Serendipity-min/image-classification-transfer-learning`，冻结提交 `3faa771c959e04b27c72e15c97d4895ed82b3cef`
- **研究负责人（PI）：**用户；**实验设计、科学审计：**ChatGPT；**代码、训练、证据及论文草稿：**Codex
- **执行范围：**仅两次 Flowers-102/ResNet-50/Full Fine-tuning/seed123/15-epoch 的 **validation-only** LR 外侧检查；随后自动进入 P4-D 论文统一重写。无新增正式五种子训练，无官方 test 评价。
- **状态：**待 Codex 接收执行；本合同不是已运行结果。

## 1. 为什么必须做、能回答什么

此前三数据集 ResNet-50 Full FT 的三档 LR `1e-5,3e-5,1e-4` 都在最高边界 `1e-4` 取得最佳。Flowers 的 P4-C1 原始候选 `records/p4/C_selected_hparams.json` 记录的是 seed123、15 epoch、第13—15轮验证正确数合计：

| 原 Flowers/R50/FT backbone LR | 第13—15轮正确数合计 / 3060 | 验证均值 Top-1 |
|---:|---:|---:|
| `1e-5` | 430 | 14.0523% |
| `3e-5` | 1772 | 57.9085% |
| `1e-4` | 2502 | **81.7647%** |

Flowers/R50/Adapter 原 P4-C1 的所选 `1e-3` 得 2777/3060 = **90.7516%**（仅同种子、早期验证参照）。封存的官方 test 主表中 R50/Adapter **89.1137%**、R50/FT **86.1604%**，差 **+2.9533 pp**。**当前差值只成立于原冻结配方，不能先验判定差异完全来自 Adapter 结构；更不能把新增 15 轮验证改善映射为旧 50 轮 test 的校正值。**

这不是纯架构控制实验。所谓“学习率按 ViT 设计”属于可能的解释，不是已证明的原合同设计意图。新实验要检验：现有 Flowers/R50/FT 的有限 LR 搜索是否在较高 LR 下还能改善早期验证表现。

## 2. 严格固定的两次运行

| 字段 | 运行 1 | 运行 2 |
|---|---|---|
| `run_id` | `P4B2-flowers102-r50-full_ft-lr1e-3-s123` | `P4B2-flowers102-r50-full_ft-lr3e-3-s123` |
| 数据 / 主干 / 策略 | Flowers-102 / `resnet50.a1_in1k` / Full FT | 相同 |
| backbone peak LR | **`1e-3`** | **`3e-3`** |
| 分类头 peak LR | `1e-2`（FT 原规则 ×10） | `3e-2`（FT 原规则 ×10） |
| 选参 seed | `123` | `123` |
| epochs | `15` | `15` |
| scheduler horizon / warmup | `50` / `5` epochs | 相同 |
| 测试集 | **禁止访问、禁止构造 test evaluator** | 相同 |

**唯一准许的科学配置变化为 backbone LR 与对应原定 ×10 的 head LR。** `run_id/stage/输出目录` 仅作身份区分。直接克隆 `P4C-tuning-flowers102-r50-full_ft-lr3-s123` 对应的可复现有效配置、代码和冻结数据身份，不重新发明配方：

- Flowers 官方 `train=1020`、`validation=1020`，102 类；**不加载 test**，不合并 train+val。
- ResNet-50 A1 的确切模型和原权重 SHA，224 输入和原 Flowers 增强、归一化；分类头 Linear(2048,102)。
- AdamW、`weight_decay=1e-4`、`betas=(0.9,0.999)`、`eps=1e-8`；cosine、原 warmup 与最小 LR 比例；参数分组与 weight decay 排除规则原样沿用。
- 物理 batch32、累积4、有效 batch128、尾窗口按真实样本数归一；每个 epoch `ceil(1020/128)=8` 次 optimizer 更新，15轮预计 **120 次更新**。
- FP32 参数与 loss、BF16 autocast、global gradient clipping norm=1、随机增强与 sampler 原确定性派生方案。
- 两个新运行都从原始已冻结预训练权重及原 seed123 分类头**重新初始化**；不得接续 P4-C1 checkpoint。P4-C1 参照头初始化 SHA：`43a9eb065534efdc6c354c984117fecfc78ffd3afde85dcc756e37a6e58f9cd5`。若实际代码重建的初始化身份不符，停止并查清，不能静默放宽。
- 不允许为高 LR 的失稳而单独改 batch、精度、warmup、权重衰减、head LR 倍率、初始化或新增梯度裁剪规则。

**已有旧网格** `1e-5/3e-5/1e-4` 原结果只读引用，不重复训练；**不补 `3e-4`**（本轮已限定两个点）。若新候选仍在边界，或 1e-3/3e-3 均不优，必须披露未搜索的 `3e-4` 等区间，不能声称充分调优。

## 3. 指标、判读与失败规则（必须在训练前冻结）

主要观察量：对于完整、数值有效的一个新候选，令

`S(LR) = 100 × (correct_epoch13 + correct_epoch14 + correct_epoch15)/(3 × 1020)`，单位为 `%`。

使用整数正确数比较，若精确同分选择较小学习率。与旧三档合并形成**五点有限搜索**；若新运行不完整或非有限，标 `numerical_failure` 或 `infrastructure_interrupted`，不得用缺失 epoch 的均值参与排序。

报告原参考 `S_old=81.764706%`，新 `S(1e-3)`、`S(3e-3)`，以及 `S_ext=max(S_old,S(1e-3),S(3e-3))`（仅有效值）。`Δ_FT_early=S_ext−S_old`。附带报告同一历史 seed123 的旧 Adapter 分数 **90.751634%** 及早期验证差，但不得据此推算正式官方 test 的新分数。

| 观察结果 | 事先决定的论文处理 |
|---|---|
| 任一新增 LR 在完整验证窗口内胜出 | 说明原 FT 三点 LR 搜索不足、原 Flowers R50/Adapter 相比 FT 的 2.953 pp **可能存在 FT 配方欠调混杂**；保留原测试数值并显著降低该排名的证据权重 |
| `Δ_FT_early ≥ 0.20 pp` | 按既有 P4 行动阈值标记明显早期 LR 敏感性；**0.20 pp 不是统计显著性阈值** |
| 新增点未胜出 | 只能说这两个新增高 LR 在 seed123、15轮窗口内未改善 FT；因 `3e-4` 等未测、50轮未知，**仍不得声称 FT 已充分调好** |
| 新最优在 `3e-3` 边界 | 记录“搜索边界仍未闭合”，不得自动继续扩大搜索 |
| 高 LR 数值失败/验证塌缩 | 原样报告失败位置、loss、grad、参数和成本，不进行按精度救援性重跑；正常基础设施中断只允许同配置合法恢复 |

最终 Flower 主测试表**仍然采用原 P4-C2 五种子固定模型**：R50/FT `86.160±0.201%` 与 R50/Adapter `89.114±0.422%`；新候选一律只看 validation，不进行 50-epoch 训练、不看 test、不替换原 C1 选择、不更新原 P4 封存。新 B2 仅作为**事后定义的学习率敏感性分析**，不能伪装为原 Flowers 预注册主比较。

## 4. 资源预算和门禁

P4-C1 原 Flowers/R50/FT 的三次 15轮 attempt 实测约 `91.79/93.90/92.01 GPU-seconds`；按约92.6秒/次外推，两个新运行基础约 **185 GPU-seconds ≈ 0.0514 GPU-hours**。这只是历史时序估计，不保证新任务实际耗时。设**预警线 0.15 GPU-hours、独立硬上限 0.50 GPU-hours（两卡时间之和）**，包括 CPU/GPU 启动、实际训练、保存、异常和必要同配置恢复；超过预警线核对剩余预测，超过硬上限拒绝派发/按退出保留停机。

- 原 P4 已使用 `9.090893/20 GPU-hours` 及其剩余 10.909107h **均不自动转账**；本 B2 用独立阶段账本和子上限，显式区分原 P4-C1/C2/C3 的成本。
- GPU0/1 各最多一个任务、严格 UUID 核对；容器资源遵循16 vCPU/64GiB，避免冲击其他训练。
- 独立新增存储规划 **4 GiB**，并保留原定物理至少14GiB的退出与原子写保护；不删除任何 P1/P2/P3/P4 模型、历史 NPZ 或原始台账。
- **原 P4 seal 存在原始 GPU 账本字节绑定失败**，仅采用 `records/p4/P4_delivery_sealed.json` 的已披露修正交付身份。不可覆盖原 `P4_sealed.json`、`gpu_ledger.final_frozen.json` 或篡改旧 attempt。B2 采用独立 `records/p4b2/gpu_ledger.json`，最终先停止一切写入，再以纯只读 API 获取 SHA、生成 `P4B2_sealed.json`；严禁用会写 `updated_monotonic` 的“只读 snapshot”在封存后修改账本。
- 服务器项目路径 **`/root/user_data/tl_cls_v1` 保持不变**，实际服务器 checkout 的科学实现 commit 不得为了交付文档而强行 `checkout 3faa771`。Codex 可在桌面/工作区从 `3faa771` 创建独立分支并只部署孤立的 B2 工具。所有新代码、配置和运行必须记录真正的执行源码 SHA、科学管线 hash、权重 hash、数据划分 hash、运行命令和 GPU 标识。

## 5. Codex 执行步骤与验收

**B2-0｜只读核对。** 确认 `3faa771` 的输入证据，P4 `C_selected_hparams.json` 旧三档正确数 `430/1772/2502`、seed123、15轮、Adapter旧参照2777；核对 P4 修正版 seal、已冻结模型和源代码身份、工作目录及无仍在运行的训练。CPU 检查不访问官方 test。

**B2-1｜最小工程实现。** 复用 `tools/p4_train.py` 和已有事务/梯度更新实现，只增加新的独立运行清单与入口，或隔离包装而不改数学管线。写入两条不可变运行记录；先 CPU 验收：参数组 LR×10、输入样本/类数、训练更新次数、确定性头哈希、采样顺序、test 访问拒绝、异常状态与事务续跑、独立账本只读封存。

**B2-2｜执行。** 运行两个15轮候选，完整记录 `epochs.jsonl`、epoch13–15 `correct/n`、数值有限检查、真实 GPU 时间、checkpoint hash 和异常。运行前不查新 test；训练中不能构造 test evaluator。常规基础设施失败可在硬上限内恢复；遇科学配置改变或数值异常，记录且不擅改配方。

**B2-3｜独立复算与封存。** 从 15 行原始 epoch 日志重新计算 `S` 与五点候选排序，核对旧 P4-C1 分数与真实新源码，写入 `P4B2_LR_validation.csv`、`P4B2_阶段结果与解释.md`、`P4B2_独立验收.json`、`P4B2_sealed.json`，审计证据最小必要项。保存两次 attempt 的终态、执行耗时及原配置不变声明。阶段无官方 test 指标、新五种子正式运行数=0。

**B2-4｜进入 P4-D 论文重写。** 两条新运行均有合法终态、可复算的完成分数或明确失效证据、独立审计与资源结算通过后，**无需再询问 PI 自动开始 Markdown 论文重写**；不得用为了让论文结论好看而重新跑/换 LR。若无法完成可信验收，应停止发布确认性 B2 结论，但可先整理论文其他已审查章节并在状态中标注阻断，不造结果。

## 6. P4-D 论文重写合同（实验完成后执行）

**输出为新稿，不覆盖原论文、不推送“最终已发表”标签；Markdown 为主格式。** 建议 `paper/论文统一重写_第一稿.md`，配套 `paper/tables/`、`paper/figures/`、`paper/claim_evidence_matrix.csv`、`paper/references.bib`、`paper/论文修改与局限清单.md`，加证据运行 manifest。必须优先阅读实际原论文文件，若仓库未收录旧 Word，请从当前受控工作区检索原稿；找不到则以已验收合同与报告撰写重构稿，并在交付说明中写清“未直接逐段比对原 Word”，禁止凭记忆还原看不见的原句。

拟定论文题目：《有限训练预算下图像分类迁移学习的精度、计算成本与概率质量权衡：跨任务实证研究》。章节：摘要、引言、相关工作、问题与实验协议、P2 选参与 P3 主实验、P4 Flowers 外部任务、P4-A/B/B2 敏感性与校准、训练/内存/存储成本、讨论与局限、结论、可复现性附录。所有表格与结果图由已验收 CSV 自动生成，保存绘图脚本、指标单位和源文件 SHA；图不能修改以掩盖负结果。文献必须核对 DOI/论文原页/GitHub 官方实现，不能虚构引用。

### 必须保留的科学结论边界

1. **原 P3 两项预注册比较**保留：CIFAR/ViT LP−FT = **−5.740 pp**、未过 −0.5pp 非劣；SVHN/R50 Adapter−FT = **+0.301 pp**、Holm p≈0.022734，但 n=5、t 分布假设敏感，不作为普适 Adapter 优势。P4-B 已证明有限 LR 搜索敏感，不能从新验证分数推断原官方 test 改写值。
2. **P4 Flowers/ViT 新主比较**：Adapter−FT = **+0.074809 pp**，95% 区间 **[−0.126413,+0.276030]**，支持预设 −0.5pp 非劣，但不等于优越或双向等效。Full FT 准确率约99.317%（错误率约0.683%），−0.5pp 的容忍降幅相当于错误率**相对可增加约73.2%**；必须在摘要/讨论或正文中说明 ceiling effect 与非劣界限的实际意义。
3. **Flowers/R50** Adapter89.114% vs FT86.160% 的 **+2.953pp** 仅为原冻结协议的描述性比较，不是 P4 的预注册优越性检验，更不能称为“R50 Adapter 机制优于 FT”。由于 FT 原 LR 网格上边界及 B2 结果，必须显著说明 **FT 欠调风险**。不管 B2 分数改善与否，均不得把 +2.953pp 当成论文的突出、确证性方法优势。
4. **CIFAR/ViT Adapter** 93.588 vs FT93.688，训练GPU-hours降低约22.6%，allocated 显存降低43.2%；这是探索性成本—精度观察，不能追认正式等效。
5. 显存 allocated ≠ 进程峰值；可训练参数数目 ≠ 全模型部署大小；ViT/ResNet 权重来源 IN21K/IN1K 混杂；Flowers 近天花板、预训练重叠未知；五训练种子与单一划分、旧公开 test 曾被观察，均须披露。
6. 已有 A1 bootstrap/McNemar、A2 ECE/Brier/NLL、A3 遮挡结果按其原统计/图像限度陈述；遮挡响应图不能作为定位或因果解释。原 P4 seal 原始字节绑定失败而采用修正交付封存、Flowers `secondary_last` 选模文本标签不精确，纳入复现性/审计边界。原始 checkpoint 和原始 NPZ 在服务器，不能自称公开了全部模型二进制或独立重推理。
7. 新 B2 只有**两次单种子早期验证 LR 补查**，在文章中作为 *post-hoc validation-only LR sensitivity*，不得称“新增高 LR 在 Flowers 官方 test 提升了 X”或“重新完成了公平 50 轮对比”。缺少中间 `3e-4` 的事实必须保留。

**提交与审查：**使用独立分支（示例 `codex/p4b2-flowers-r50ft-and-paper`），建议分两次提交：`B2 evidence and terminal seal`，随后 `P4-D full manuscript draft with provenance`。推送 GitHub 后回交两次 commit、文档路径、全部新实验结果与 GPU 台账；Codex 只生成论文**供 PI 审核的第一稿**，不自动合并 main、不自动投稿、不启动其他 GPU 实验。

## 7. 必交回执与阻断条件

- 文件：`reports/P4B2/P4B2_LR_validation.csv`、`reports/P4B2/P4B2_实验结果与分析.md`、`records/p4b2/acceptance.json`、`records/p4b2/P4B2_sealed.json`、`records/p4b2/gpu_ledger.json`（导出副本）、`paper/论文统一重写_第一稿.md`、`paper/claim_evidence_matrix.csv` 和复现/作图脚本。
- 阶段签收条件：两个 LR/run_id 均有终态；完成者15/15 epoch、对应120次 optimizer 更新、数据来源/分类头初始化 SHA 一致、测试集未访问、原模型/权重/原 P4 seal 不变、账本记录完备并独立复算、额度不越界。
- 硬阻断：对新候选调用 test；修改 P3/P4 的已封存官方测试表；篡改既有 seal；自动为数值失败改变科学配方；超过 0.50 GPU-hour 硬上限；无法证明配置身份；出现未解释的样本划分冲突。失败也必须保留，不允许把它从分母中删除。

## 8. 一段可以直接粘贴给 Codex 的指令

> 以 `Serendipity-min/image-classification-transfer-learning@3faa771c959e04b27c72e15c97d4895ed82b3cef` 为只读审计基线，执行本合同 `TL-CLS-P4B2-20261028-v1.0`。仅为 Flowers-102/ResNet50-A1/Full-FT/seed123 新增 backbone LR=`1e-3` 和 `3e-3` 两个 **15 epoch validation-only** 候选，FT 分类头始终×10，50-epoch scheduler horizon和5-epoch warmup不变，完全复用 P4C1 冻结实现/训练数据/增强/参数组。零新 test 推理，零新增五种子50轮训练；旧 P2/P3/P4 指标、checkpoint、seal 只读。新账本独立，硬上限 **0.50 GPU-hours**，预测告警0.15h。按epochs13–15正确数复算五点LR敏感性，记录数值失败和未搜索 `3e-4` 的局限，修正封存的旧缺陷仍披露而非覆盖。完成且通过独立验收后**自动进入 P4-D**：使用P2/P3/P4和本次B2全部可信结果，重写完整中文Markdown论文并生成结果表图/证据矩阵/真实参考文献；明确Flowers/R50 Adapter +2.953pp可能受FT欠调影响、Flowers/ViT 99.317%天花板与0.5pp非劣界≈73.2%相对错误率容忍、原P3两项主比较保持不变。新分支先提交B2证据，再提交论文第一稿并推送；不合并、不投稿、不再自动扩实验，回交 commit和PI审核入口。

## 9. 一手证据固定链接

- [P4 最终数据交付与原 seal 例外](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/3faa771c959e04b27c72e15c97d4895ed82b3cef/reports/P4/数据交付与验收.md)
- [Flowers 8配置×3 LR 选参原始记录](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/3faa771c959e04b27c72e15c97d4895ed82b3cef/records/p4/C_selected_hparams.json)
- [Flowers 五种子结果（含GPU时间）](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/3faa771c959e04b27c72e15c97d4895ed82b3cef/reports/P4/C_configuration_summary.csv)
- [P4 修正交付封存](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/3faa771c959e04b27c72e15c97d4895ed82b3cef/records/p4/P4_delivery_sealed.json)
- [P2 选参和 P3 官方主比较](https://github.com/Serendipity-min/image-classification-transfer-learning/blob/2b19bedca2b13b6ee3d2ebd2e0b8ba244722ca6d/reports/P3_C_完整结果与验收.md)
