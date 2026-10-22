"""仅用已核验终态证据生成中文回执和 P3 待审提案；不会修改执行授权。"""

import argparse
import csv
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))

    report = read('reports/P2_report.json')
    audit = read('records/p2/delivery_audit.json')
    selection = read('selected_hparams.json')
    proposal = read('reports/P3_proposal.json')
    assert audit['status'] == 'passed' and selection['all_configurations_selected']
    assert not proposal['P3_authorized'] and not proposal['centralized_test_authorized']
    ledger = read('records/p2/gpu_ledger.json')
    attempts = sorted(ledger['attempts'].values(), key=lambda a: a['started_monotonic'])
    storage = report['storage']
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    actual = report['P2_actual_gpu_hours']
    count = report['P2_completed_candidates']
    numerical = report['P2_numerical_failures']
    checkpoint = read('records/p2/checkpoint_inventory.json')
    lines = ['# P2 阶段验收回执', '',
        f'交付生成时间：{date}。执行提交 `{report["execution_commit"]}`；归档提交见 Git 历史。', '',
        f'原矩阵 48 个候选已到终态：完整完成 {count}，数值失败 {numerical}；16 个配置的 LR 选择均已冻结。'
        f'实际 P2 成本 {actual:.4f} / 30 GPU-hours，所有验收、技术前缀和重跑均计入。P3 与集中测试保持关闭。', '',
        '## 工程与科学身份', '',
        'seed=123；每候选 15 个完整 epoch；调度 horizon=50、warmup=5；micro-batch=32、累积=4、有效 batch=128。'
        'BF16 autocast、FP32 参数/loss、TF32 关闭；初始化、冻结权重、数据 split、增强和优化规则沿用原协议。', '',
        f'真实 CPU/GPU 入口验收与执行 commit、文件 SHA 和授权 SHA 绑定。{read("records/p2/cpu_acceptance.json")["tests_run"]} 项 CPU 检查通过；真实 GPU 检查的恢复 logits 差为 0，'
        '继续更新后完整模型及 optimizer 逐位相同，改变 LR 的恢复被拒绝。四份更新/数据/模型/随机流数学源码与已接受 P1 版本逐字相同。', '',
        '[组合验收](../records/p2/acceptance.json) · [CPU 验收](../records/p2/cpu_acceptance.json) · '
        '[GPU 验收](../records/p2/gpu_acceptance.json) · [独立交付复核](../records/p2/delivery_audit.json)', '',
        f'独立复核重新核对 {audit["manifest_files_verified"]} 个原始证据文件、事务链、'
        f'{audit["candidate_counts"]["epochs"]} 个有效完整 epoch、'
        f'{audit["candidate_counts"]["samples"]:,} 个训练样本和 {audit["candidate_counts"]["updates"]:,} 次 optimizer 更新。'
        '逐轮配对 sampler 与分类头初始化 SHA 一致；尾窗口按实际样本数计算。', '',
        '## 全部 LR 候选与选择', '',
        '下表为验证 Top-1 百分数；每档数值是第 13–15 轮的均值。程序用三轮正确数总和精确比较，并列取较小数值 LR。'
        '数值失败保留状态，不以未完成均值参加选择。P2 单种子结果仅用于调参，不构成五种子正式方法结论。', '',
        '| 配置 | 三档 LR：验证 Top-1 均值 (%) | 所选 LR |', '|---|---|---:|']
    for name, chosen in sorted(selection['configurations'].items()):
        values = []
        for candidate in sorted(chosen['candidates'], key=lambda c: c['learning_rate']):
            score = f'{candidate["selection_value"]:.6f}' if candidate.get('eligible') else candidate['status']
            values.append(f'{candidate["learning_rate"]:g}: {score}')
        lines.append(f'| {name} | {"；".join(values)} | {chosen["learning_rate"]:g} |')
    lines += ['', '[完整 48 候选 CSV](P2_candidate_results.csv) · [全部原始 epoch CSV](P2_epoch_curves.csv) · '
        '[选参 JSON](../selected_hparams.json) · [16 组曲线与图源](P2_figures/)', '',
        '## 成本、校准与技术重试', '',
        f'共享父账本累计 {actual:.6f} GPU-hours，剩余 {30-actual:.6f}；'
        f'{len(attempts)} 个 attempt 已全部结算。两卡重叠按两份占用相加，启动、半轮失败、保存和退出均不扣除。'
        '学校服务器无计价，金额项不适用。', '',
        '| 首批完整第 2–3 轮校准 | 实测秒/epoch | P1 基线秒/epoch | 比值 |', '|---|---:|---:|---:|']
    first = read('records/p2/candidate_plan.json')['first_calibration_candidates']
    for item in report['cost_calibration']['observations']:
        if item['run_id'] in first:
            lines.append(f'| {item["run_id"]} | {item["epoch2_3_fit_seconds"]:.3f} | '
                         f'{item["P1_epoch2_3_baseline_seconds"]:.3f} | {item["slowdown_ratio"]:.5f} |')
    flags = report['cost_calibration']['above_10_percent_run_ids']
    lines += ['', f'完整阶段校准中超过 10% 调查线的候选数：{len(flags)}。'
        '全部调查与调度调整如发生，见阶段记录和操作覆盖记录；不改变 LR、seed、训练轮数或精度。', '',
        '初版执行代码 `092bcfb` 的两个未完成前缀因终轮摘要恢复和协调退出分类修正而归档；'
        '旧代码、验收、日志、模型身份及实际费用保留。修正后 `6c0a215` 重新验收并全新执行同一登记候选。'
        '该技术重跑不以得分改善为目的，旧前缀不参与选择。', '',
        '另有存储统计原子重命名竞态中断，已在 `d66193b` 修复。17 个完成项保持原身份，两个中断前缀按显式运维兼容映射恢复；'
        '真实运行代码身份与冻结科学基线分开保存，不能把旧摘要的执行 commit 改为新 commit。'
        '具体原因、全部备份 SHA 和恢复验收见 [存储事故恢复记录](../records/p2/storage_race_recovery.json)。', '',
        '[共享 GPU 台账](../records/p2/gpu_ledger.json) · [逐 attempt 表](P2_attempts.csv) · '
        '[技术修订](../records/p2/technical_revisions.json) · [全阶段校准](../records/p2/cost_calibration.json)', '',
        '## 存储与服务器模型', '',
        f'项目 {storage["project_bytes"]/2**30:.3f} GiB，独立环境非链接文件 '
        f'{storage["environment_non_symlink_bytes"]/2**30:.3f} GiB，合计 {storage["combined_bytes"]/2**30:.3f} GiB。'
        f'用户确认分配 {storage["confirmed_allocation_bytes"]/2**30:.0f} GiB，余 '
        f'{storage["allocation_remaining_bytes"]/2**30:.3f} GiB；仍保留至少 5 GiB 退出余量。'
        '这里报告逻辑文件字节，排除符号链接并在每个根目录内去重硬链接；共享盘空闲未冒充租户配额。', '',
        f'P1 的 40 份 checkpoint 全部保留并在服务器重新 SHA 核验；P2 现存索引含 '
        f'{len(checkpoint["P2_files"])} 份模型文件（含技术归档现场）。best、resumable last 和 LP/Adapter 含 buffers 的任务增量均留服务器。'
        '完整事务后回收的 superseded 中间代次保留 SHA 历史，不冒充现存模型。', '',
        '工作目录 `/root/user_data/tl_cls_v1`，环境 `/root/miniconda3/envs/tl_cls_v1`，路径保持不变。'
        '本次只回收代码、日志和回执，checkpoint 传输数为 0。', '',
        '[现存 checkpoint 清单](../records/p2/checkpoint_inventory.json) · [原始证据清单](../records/p2/evidence_manifest.json)', '',
        '## 后续审核', '',
        '[P3 阶段审核提案](../docs/P3_阶段审核提案.md) 与 [80-run 解析配置及预算 JSON](P3_proposal.json) 已准备。'
        '它们均为未授权草案，P3、集中测试、测试预测保持关闭；等待下一次 PI 阶段审核。', '']
    (root / 'reports/P2_验收回执.md').write_text('\n'.join(lines), encoding='utf-8')
    p3 = ['# P3 阶段审核提案（未授权）', '',
        'P2 证据已冻结，本文件仅提出下一阶段资源和实现要求；不启动 P3 或测试。', '',
        '16 个所选配置 × 原定五种子 17、42、2026、3407、65537 = 80 个正式 run；'
        '全部从原始初始化重新训练 50 轮，不续训 P2。所选 LR、P2 run 和 selected_hparams SHA 已解析进入配置草案。', '',
        '| 预算组成 | GPU-hours 提案 |', '|---|---:|',
        f'| 80-run 基础训练外推 | {proposal["P3_base_training_gpu_hours"]:.3f} |',
        f'| 最多 240 个额外时间切片保存情景 | {proposal["P3_extra_prefix_save_gpu_hours_upper_scenario"]:.3f} |',
        f'| 集中评价上界情景 | {proposal["centralized_evaluation_gpu_hours_upper_scenario"]:.3f} |',
        f'| 诊断预留 | {proposal["diagnostic_reserve_gpu_hours_proposal"]:.3f} |',
        f'| 推理基准预留 | {proposal["inference_benchmark_reserve_gpu_hours_proposal"]:.3f} |',
        f'| 上述合计加 25% | {proposal["P3_training_prefix_evaluation_reserves_with_25_percent_margin_gpu_hours"]:.3f} |', '',
        '外推采用各配置所选 P2 候选第 2 轮起的完整 epoch 事务时间中位数，加实际固定和首轮额外开销。'
        'P2 实测已包含当时并发环境，不再次重复乘同一个慢化因子。它是估算，须在正式运行中继续按共享台账执行新的独立硬上限。', '',
        '额外切片保存按每个 run 最多三次完整事务费用保守计；评价按 80 best + 80 last + 最多 240 prefix，'
        '用 P2 验证时序按官方测试集样本数缩放，尚未执行任何测试评价。重复 checkpoint 的预测可去重复用，草案未预先据此压低上界。', '',
        '## 共同时间预算草案', '',
        '每个 dataset/backbone 组的 B 建议取四方法 timing-only 50 轮估算的最小值，切片为 0.25B、0.5B、B。'
        '这属于使用 P2 时序校准原 P1 时间公式，须由 PI 冻结；不得依据观察到的准确率改选 B。', '',
        '| 组 | B 秒 | 0.25B 秒 | 0.5B 秒 |', '|---|---:|---:|---:|']
    for name, item in sorted(proposal['common_time_slices'].items()):
        b = item['B_seconds_proposal']
        p3.append(f'| {name} | {b:.3f} | {b*.25:.3f} | {b*.5:.3f} |')
    size = proposal['project_environment_full_retention_with_prefix_and_25_percent_margin_bytes'] / 2**30
    p3 += ['', '## 存储与进入条件', '',
        f'保留当前 P1/P2 全证据、P3 best/last/delta、每 run 三个额外完整切片并加 25% 的存储情景约 '
        f'**{size:.2f} GiB**。本次 60 GiB 仅覆盖已授权 P2；此新情景需要 PI 明确资源和保留政策，不能自动沿用为已获保障。', '',
        'P3 入口验收必须覆盖 80-run 配置、不可变 prefix 提交及时间冻结、P3 seal 与 evaluation manifest，'
        '以及唯一 evaluation_id、run+checkpoint SHA 预测复用和 TestData 构造前的严格身份/状态/角色/预算时刻验证。'
        '原评价入口对多个 prefix 的命名覆盖问题须在开放测试前解决。', '',
        '每个预算点只接受预算内已完成验证并提交保存的 checkpoint；无合格模型就报告不可用。'
        '正式失败保留五种子分母，主比较、配对分析、置信区间、Holm 校正及非劣效界限沿用原合同。', '',
        '[机器可读预算及 80 个解析配置](../reports/P3_proposal.json) · [完整 P2 回执](../reports/P2_验收回执.md)', '']
    (root / 'docs/P3_阶段审核提案.md').write_text('\n'.join(p3), encoding='utf-8')
    # 派生 CSV 使用父台账原值；终值 RSS 与采样峰值保持不同列名。
    keys = ['attempt_id', 'run_id', 'kind', 'device', 'execution_commit', 'status', 'returncode',
        'termination_reason', 'started_unix', 'charged_gpu_seconds', 'main_process_peak_rss_bytes_sampled',
        'process_tree_peak_rss_bytes_sampled', 'samples']
    with (root / 'reports/P2_attempts.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: a.get(key) for key in keys} for a in attempts)
    print(json.dumps({'P2_receipt_written': True, 'P3_proposal_written': True, 'P3_authorized': False}))


if __name__ == '__main__':
    main()
