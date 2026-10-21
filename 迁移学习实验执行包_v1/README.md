# 图像分类迁移学习实验执行包

协议：TL-CLS-20261020-v1.0  
日期：2026-10-20  
状态：实验设计和待执行清单，未运行服务器实验。

## 使用顺序

1. 阅读 `实验设计与执行合同.md`，以其科学定义、分析范围和首轮任务为准。
2. Codex 将 `protocol.json` 转化为项目实际配置，实现数据、模型、策略、训练和评价入口。
3. 读取 `run_matrix.json` 的 `planned_runs`，不要从已有成功日志反推运行清单。
4. 首次执行 P0–P1，建议上限 4 GPU-hours；回交实现验收、16 配置各 3 epochs 的日志及后续预算。
5. 用真实 P0 证据补齐依赖、权重 revision/hash、split hash；P1 后冻结 P2/P3 的实际预算。
6. P2 只用验证集选择学习率，写出 `selected_hparams.json`；P3 按引用解析该文件，从头执行 80 次正式训练。
7. 所有配置、分析规则和运行状态冻结后，集中评价合法 checkpoint；按 `result_record.template.json` 输出证据。

此包提供研究规格与运行清单，不是已经实现的训练框架；不能直接将 JSON 文件当成训练脚本启动。实现工作由 Codex 完成。

## 文件说明

| 文件 | 内容 |
|---|---|
| `实验设计与执行合同.md` | 完整审查、实验设计、超参数、NaN 诊断、统计与验收要求 |
| `protocol.json` | 机器可读的实验规格；未知运行事实保留 null |
| `run_matrix.json` | P1 16 次、P2 48 次、P3 80 次，共 144 条 planned 记录 |
| `result_record.template.json` | 每个 run 的结果字段模板；所有实测指标初始为 null |
| `reference/预实验原稿.docx` | 用户原始文档的未改动副本，仅用于历史核对 |
| `CHECKSUMS.sha256` | 包内交付内容的校验值，不包含本校验文件自身 |

## 运行数和阶段边界

| 阶段 | 配置/运行数 | 每次 epochs | 总 epochs |
|---|---:|---:|---:|
| P1 | 16 | 3 | 48 |
| P2 | 48 | 15 | 720 |
| P3 | 80 | 50 | 4,000 |
| 合计 | 144 | 不适用 | 4,768 |

80 次 P3 正式运行才进入五种子主结果；P1/P2 不计为额外正式种子。历史 NaN 诊断、并发微基准、推理评价、10% 数据和 ViT IN1K 扩展不包含在 144 条训练清单中，费用另列。

## 关键待填字段

- P0：实际环境版本、代码 commit、数据与权重 SHA-256、权重 revision、split manifest、单卡显存与容器配额。
- P1 后：阶段总 GPU-hours、项目费用和存储预算。25% 余量只是预算建议，不是已发生的计算。
- P2 后：各 configuration_id 的选定学习率，以及 P3 共同时间切片的秒数。
- P3 后：运行状态、测试成绩、资源使用与逐样本预测。

`null` 不表示 0、通过或无故障。不得用论文旧成绩填入新结果。

## selected_hparams 文件的最小结构

下列片段展示结构，不是已经选择的超参数；实际生成时必须包含全部 16 个配置及候选证据。

~~~json
{
  "protocol_id": "TL-CLS-20261020-v1.0",
  "configurations": {
    "cifar100-r50-adapter": {
      "learning_rate": null,
      "head_lr_multiplier": 1.0,
      "selected_from_run_id": null,
      "selection_metric": "mean_val_top1_epochs_13_to_15",
      "selection_value": null
    }
  }
}
~~~

P3 的引用形式为 `selected_hparams.json#/configurations/<configuration_id>/learning_rate`。读取后检查不是 null、来自本轮合法候选且与选择规则一致，才能训练。

## 不应混淆的结果

`completed`、`numerical_failure`、`stalled`、`oom`、`infrastructure_interrupted`、`invalid_protocol`、`blocked` 与 `cancelled` 各自保留。正式主比较需要双方五种子均完整合法；恢复 checkpoint 单列。

测试结果不用于修改本轮冻结的选择规则。根据结果产生的下一轮研究问题可以登记为独立扩展，由用户调整方向；不得重新包装成原先预注册的假设。
