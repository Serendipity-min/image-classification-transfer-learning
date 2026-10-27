> P3-C：PI已明确授权并取消独立签署，本轮28项CPU验收通过，已后台启动冻结400项评价，3/160 GPU-hours上限不变；完成后回收分析推送。详见[直接授权与启动回执](reports/P3_C_PI直接授权与后台启动.md)。历史待签署回执只代表当时状态。

# 图像分类迁移学习：P3-B 完整交付

2026-10-27：**80/80 正式运行完成，4,000 epochs，独立验收 PASS，已封存。P3-C 与官方集中测试保持关闭。** 仓库已由用户设为公开，`codex/p3-execution` 启用 GitHub 平台保护。

- [完整结果汇报与验收](reports/P3_完整结果汇报与验收.md)、[服务器逐run交接](reports/P3_阶段交接报告.md)、[独立验收JSON](records/p3/delivery_audit.json)、[P3 seal](records/P3_sealed.json)。
- [五seed验证结果](reports/P3_validation_configurations.csv)、[80-run结果](reports/P3_validation_runs.csv)、[4,000轮曲线源数据](reports/P3_epoch_curves.csv)、[图表与可编辑源图](figures/P3)。所有准确率均为验证集，未运行官方测试。
- 实际累计 **124.610609507 / 160 GPU-hours**；92个attempt全部结算，含4次暂停和8次GPU工程验收。P1/P2证据完整保留，全部checkpoint留在原服务器路径 `/root/user_data/tl_cls_v1`。
- 科学基线 `7247c661`；实际逐轮执行为 `7247c661`（2,740轮）和 `700a1adc`（1,260轮）。54个原终态summary未改写，4次合法同run恢复；归档commit与训练commit分开。
- [完整文本证据](evidence/P3_final_20261027)、[GPU台账](records/p3/gpu_ledger.json)、[checkpoint清单](records/p3/checkpoint_inventory.json)、[评价manifest草案](records/p3/evaluation_manifest.draft.json)。
- [交付收尾回执](records/p3/delivery_closeout.json)：服务器定时监控已结束，桌面30分钟定时任务已暂停；集中测试保持关闭。
- [下一阶段预算提案](reports/P3_C_待审预算.json)：建议集中评价子预算3 GPU-hours，**待用户单独授权**；不自动启动测试、扩展实验或论文定稿。
- 历史过程：[启动验收](reports/P3_启动验收回执.md)、[首次暂停与恢复](reports/P3_暂停原因与恢复验收.md)、[65-run中期调查](reports/P3_65run中期结果与吞吐调查.md)、[P3合同](docs/P3_有条件执行合同.md)。历史状态仅代表当时快照。

P2历史报告：[阶段PDF](reports/P2_阶段汇报.pdf)、[结果分析](reports/P2_结果汇报.md)、[完整验收](reports/P2_验收回执.md)、[冻结LR](selected_hparams.json)。

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
