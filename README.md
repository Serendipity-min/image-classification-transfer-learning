# 图像分类迁移学习：P2 完成与 PI 审核

协议 `TL-CLS-20261020-v1.0`，更新日期 2026-10-22（北京时间）。

**P2 已完成并通过工程与选参验收：48/48 候选、720 个完整 epoch、16 组所选 LR；实际 22.1117 / 30 GPU-hours。P3 与集中测试保持关闭。**

- [5 页阶段汇报 PDF](reports/P2_阶段汇报.pdf)、[中文分析汇报](reports/P2_结果汇报.md)、[完整验收回执](reports/P2_验收回执.md)。
- [16 组全部曲线](reports/P2_figures/)、[三档 LR 比较](reports/P2_LR_comparison.csv)、[48 候选结果](reports/P2_candidate_results.csv)、[选参 JSON](selected_hparams.json)。
- [P3 提案](docs/P3_阶段审核提案.md)：全包情景约 156.83 GPU-hours、182.80 GiB；建议单独审核 160 GPU-hours、至少 200 GiB。尚未授权。
- [独立交付复核](records/p2/delivery_audit.json)、[GPU 台账](records/p2/gpu_ledger.json)、[存储采样台账](reports/P2_storage_ledger.csv)、[checkpoint SHA 清单](records/p2/checkpoint_inventory.json)。

17 个候选保留 `6c0a215` 原运行身份，31 个候选以 `d66193b` 完成或恢复。全部实际费用、两次工程事故与修复均留痕。模型继续保存在服务器，本地与 GitHub 仅回收代码、配置、日志、曲线和回执。

## P2 当前入口

- [启动验收与技术修订回执](docs/P2_启动验收回执.md)、[执行计划](docs/P2_执行与验收计划.md)、[PI 补充审查](docs/P2_阶段启动审查与Codex执行补充_v1.md)。
- [真实启动验收](records/p2/acceptance.json)、[18 项 CPU 检查](records/p2/cpu_acceptance.json)、[GPU 事务恢复检查](records/p2/gpu_acceptance.json)。
- [P2 授权及资源记录](records/p2/authorization.json)、[技术修订留痕](records/p2/technical_revisions.json)。

初始执行提交为 `6c0a215798a4de319e7c6c55f0bb1b351bf082e0`，恢复执行提交为 `d66193bc3265954094e0badebfcd3279deb69e5a`，位于 `codex/p2-execution` 分支。运行代码只在 `/root/user_data/tl_cls_v1` 部署，独立预算硬封顶 30 GPU-hours，涵盖验收、失败、技术重跑、恢复和退出。用户确认原项目及环境至少 60 GiB 分配；54 GiB 预警，至少保留 5 GiB 退出写入余量。

原矩阵为 48 个候选，seed123，每次 15 轮，50 轮调度、5 轮 warmup。按第 13–15 轮验证 Top-1 均值选 LR，并列小 LR。完整第 2–3 轮校准、超过 10% 的慢化调查以及预算/存储保护均已启用。完成全部候选及 P3 提案后停止，不自动升级 P3。

## 已接受的 P0/P1 证据

1. [PI 审核与阶段分析](docs/PI_审核与阶段分析.md)：已知事实、证据局限、后续预算和待裁定事项。
2. [P0–P1 验收回执](reports/P0_P1_验收回执.md)：环境、数据/权重身份、实现检查和 16 组实测结果。
3. [完整配置结果表](reports/P1_acceptance_table.csv)、[结构化总回执](reports/P0_P1_report.json)、[验收审计](records/acceptance_audit.json)。
4. [预算补充](reports/acceptance_addendum.json)、[分配置预算及两卡排程提案](records/budget_plan.json)。
5. [PI 决定模板](docs/PI_审核决定模板.md)、[机器可读阶段状态](records/pi_review_status.json)。

## 已接受的 P0/P1 工程数据（历史）

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

## 历史预算与下一次审核

历史 P1 回执中的 21.50 GPU-hours 是 P2 独占基线。乘代表并发情景后约 23.20，再加 25% 约 29.00；用户现已授权独立 30 GPU-hours 硬上限。历史 179.59 GPU-hours 尚未完整计入并发与切片，不能直接当作 P3 全包额度。学校服务器无计价，金额项不适用。

P2 初始完整保留方案约 42.42 GiB、加 25% 约 53.02 GiB；至少 60 GiB 分配已获用户确认。技术前缀额外文件也计入实际监控。历史 117.32/180.45 GiB 属整个 P1/P2/P3 及切片情景，P3 将以 P2 证据重新提案，等待下一次审核。

## 证据与来源

- `src/`、`configs/`：实际实验实现和冻结配置。
- `runs/<run_id>/`：每组解析配置、三轮日志、资源及 checkpoint hash；checkpoint 二进制留服务器。
- `records/`：环境、数据/权重身份、split、实现验收、GPU 台账、40 份 checkpoint 核验索引。
- `reports/figures/`：四组曲线、全部 48 个 epoch 图源、PDF/SVG/PNG 和版面检查。
- [原始证据包](reports/P0_P1_验收证据包.zip)、[交付清单](reports/delivery_manifest.json)、[服务器证据清单](records/evidence_manifest.json)。P1 清单对应固定 `b485177` 交付快照；新阶段源码不回填旧清单。`tools/verify_handoff.py` 在该历史 Git 快照核验，P2 则用 `tools/verify_p2_execution.py`。
- `迁移学习实验执行包_v1/`：原始设计和历史稿件输入，保留其制定时的状态。
- `docs/history/`：历史审查材料；其中推断的更正见本次 PI 分析，不作为新的实测结论。

训练源码身份为 `74001fc68c2bfb68cd65da821c961d75a93fe32b`（前六组）和 `94b02df2ae94097602e40e88783f8c445006fa9a`（其余十组）。两者仅调度器不同，训练/模型/数据源码一致。仓库保留这两个原始 commit；本次归档提交不能回填成历史 run 的源码身份。

服务器工作目录 `/root/user_data/tl_cls_v1`；项目环境 `/root/miniconda3/envs/tl_cls_v1`。仓库不提供连接凭据。P1 旧入口保持原阶段限制，P2 使用独立入口；集中测试已在构造 TestData 前明确拒绝本阶段调用。归档提交不能替代各 run 的真实执行提交。
