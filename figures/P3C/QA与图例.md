# P3-C 图例与图形验收

Python/matplotlib 生成 PDF/SVG 和 600 dpi PNG；宽 180.34mm，配对图高 96.52mm，另外两图高 147.32mm。三图的对齐报告均 PASS、无豁免；三份 PDF 的最小字形均 7pt，字体可编辑；渲染碰撞均为 0 FAIL / 0 WARN。源代码审查为 20 PASS / 1 WARN / 0 FAIL：唯一 WARN 为没有 TIFF，本交付采用矢量 PDF/SVG、PNG 仅为预览，不声称提交 TIFF，因此接受此提示。

图形只用已验收的真实 CSV。所有配置和五 seed 均保留，无数据筛选、合成、平滑或插值。主图为均值±样本 SD；配对图为差值均值及 t 95% CI（df=4），不能混读。颜色始终对应同一种方法，方向及类别另用位置、标记、文本标明。

| 图 / 面板 | 独有证据作用 | 中心 / 区间及单位 | 文字、间距与可见性复核 | 裁定 |
|---|---|---|---|---|
| primary / a | CIFAR/R50 的方法排序边界 | 五 seed 全点；均值±SD | 四方法标签完整；同轴 0–100；误差线虽短但保留 | PASS |
| primary / b | CIFAR/ViT 的迁移与 Scratch 边界 | 五 seed 全点；均值±SD | IN21K 标明；不把近似点位当等效 | PASS |
| primary / c | SVHN/R50 的 Scratch 高分边界 | 五 seed 全点；均值±SD | Scratch/Adapter/FT 接近；精确数值见表 | PASS |
| primary / d | SVHN/ViT 的 LP 损失边界 | 五 seed 全点；均值±SD | 四策略全部显示；不隐藏低分方法 | PASS |
| paired / a | 预注册 LP 非劣失败 | 五配对差；均值及 95% CI | 零与 −0.5 界线明确；区间未遮挡；x 轴独立明示 | PASS |
| paired / b | 预注册 Adapter 相对 FT 正效应 | 五配对差；均值及 95% CI | 零线与差值方向清楚；没有跨面板共用不适当尺度 | PASS |
| prefix / a | CIFAR/R50 的预算依赖排序 | 五 seed 聚合均值±SD | 3 个预算点均保留；方法颜色一致 | PASS |
| prefix / b | CIFAR/ViT 的 LP 代价及 Adapter 取舍 | 五 seed 聚合均值±SD | FT/Adapter 接近，CSV 保留精确值；不作正式等效声明 | PASS |
| prefix / c | SVHN/R50 的早期与完整预算边界 | 五 seed 聚合均值±SD | 无截断；小误差条没有省略 | PASS |
| prefix / d | SVHN/ViT 的任务依赖边界 | 五 seed 聚合均值±SD | Scratch 不确定性可见；3 截点连接只为视觉引导 | PASS |

已分别查看三幅完整成图的全部 10 个面板；无文字截断、越界或相互遮挡。放大主要配对图而非用均值图推断显著性。全范围主图用于方法概况，数值接近的比较由精确表格与预注册差值图承担；未用截短百分比轴放大微小优势。

图例：primary_test_results 的点为各固定训练 seed 在冻结 validation-best checkpoint 上的官方 test Top-1，黑线为均值±样本 SD；preregistered_paired_contrasts 的点为同 seed 下 A−FT 差值，黑色菱形和线为均值与配对 t 95% CI，灰虚线为零、橙点线为 −0.5pp 非劣界；fixed_budget_test_prefixes 为相同训练轨迹在冻结共同 B 截点的官方 test 均值±SD，240 个角色全部进入汇总。全部 n=5 个训练 seed。前缀非独立实验，两主比较 Holm p 值在主报告，其他结果均探索性。

源数据：`reports/P3_C_逐角色测试指标.csv`、`P3_C_主比较差值.csv`、`P3_C_逐种子配对统计.csv`。脚本：`tools/plot_p3c_results.py`。自动结果见 `automated_qa.json`、各图 alignment/collision JSON；collision-overlay PDF 仅作诊断，不替代正式图。
