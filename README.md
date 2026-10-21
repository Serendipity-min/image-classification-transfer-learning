# 图像分类迁移学习：P0–P1 验收与 PI 审核

协议 `TL-CLS-20261020-v1.0`，更新日期 2026-10-21（北京时间）。

**当前状态：P0 通过，P1 完成，等待 PI 审核。P2/P3 未启动、未授权。**

## 审核入口

1. [PI 审核与阶段分析](docs/PI_审核与阶段分析.md)：已知事实、证据局限、后续预算和待裁定事项。
2. [P0–P1 验收回执](reports/P0_P1_验收回执.md)：环境、数据/权重身份、实现检查和 16 组实测结果。
3. [完整配置结果表](reports/P1_acceptance_table.csv)、[结构化总回执](reports/P0_P1_report.json)、[验收审计](records/acceptance_audit.json)。
4. [预算补充](reports/acceptance_addendum.json)、[分配置预算及两卡排程提案](records/budget_plan.json)。
5. [PI 决定模板](docs/PI_审核决定模板.md)、[机器可读阶段状态](records/pi_review_status.json)。

## 本轮完成情况

| 项目 | 实测结果 |
|---|---|
| P0 配置验收 | 16/16 通过，含参数白名单、BN、Adapter 恒等/梯度、累积尾批和恢复检查 |
| P1 短程训练 | 16/16 × 3 完整 epoch，共 48 epoch |
| GPU 预算台账 | 1.5234 / 4 GPU-hours |
| 样本处理 / optimizer 更新 | 2,662,344 / 20,832 |
| OOM / 非有限值 | 本轮未观察到 |
| 测试集模型评价 | 0 次 |
| 代表任务并发减速 | 7.9%、7.1%；未超过 10% 调查阈值 |

P1 是一个种子的三轮 profile，仍处于五轮 warmup 内。它用于工程验收和资源估算，不能代替五种子正式结果，也不能证明 50 轮稳定或确认历史 NaN 根因。

## PI 需要审核的资源

P2 训练约 21.50 GPU-hours，P3 训练约 119.08 GPU-hours；训练加 25% 余量约 175.73 GPU-hours。另计集中评测估计、诊断和推理预留后，整体提案约 179.59 GPU-hours；尚未冻结的预算切片费用另计。两卡训练排程含争用情景及余量约 94.92 小时。学校服务器无计价，金额项不适用。

当前项目约 11.37 GiB。P1/P2/P3 全部保留 best、last、任务增量并加 25% 存储余量约 117.32 GiB；每次正式运行再保留三份完整切片权重的情景约 180.45 GiB。租户真实配额仍未确认，不能将共享盘剩余空间视为项目额度。建议先审核 P2，再根据选参结果、切片方案和存储政策单独决定 P3。

## 证据与来源

- `src/`、`configs/`：实际实验实现和冻结配置。
- `runs/<run_id>/`：每组解析配置、三轮日志、资源及 checkpoint hash；checkpoint 二进制留服务器。
- `records/`：环境、数据/权重身份、split、实现验收、GPU 台账、40 份 checkpoint 核验索引。
- `reports/figures/`：四组曲线、全部 48 个 epoch 图源、PDF/SVG/PNG 和版面检查。
- [原始证据包](reports/P0_P1_验收证据包.zip)、[交付清单](reports/delivery_manifest.json)、[服务器证据清单](records/evidence_manifest.json)。清单覆盖对应交付快照；新增 PI 文档由 Git 记录。
- `迁移学习实验执行包_v1/`：原始设计和历史稿件输入，保留其制定时的状态。
- `docs/history/`：历史审查材料；其中推断的更正见本次 PI 分析，不作为新的实测结论。

训练源码身份为 `74001fc68c2bfb68cd65da821c961d75a93fe32b`（前六组）和 `94b02df2ae94097602e40e88783f8c445006fa9a`（其余十组）。两者仅调度器不同，训练/模型/数据源码一致。仓库保留这两个原始 commit；本次归档提交不能回填成历史 run 的源码身份。

服务器工作目录 `/root/user_data/tl_cls_v1`；项目环境 `/root/miniconda3/envs/tl_cls_v1`。仓库不提供连接凭据；连接运行时不随交付发布。现有训练入口仅接受 P1，集中测试入口要求 P3 状态封存。PI 审核和后续阶段授权完成前，不执行新的训练或测试。
