"""以冻结回执生成中文阶段汇报（Markdown/PDF），重新计算资源提案，禁止读取测试预测。"""

from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['scratch', 'full_ft', 'linear_probe', 'adapter']
LABELS = {'scratch': 'Scratch', 'full_ft': 'Full FT', 'linear_probe': 'LP', 'adapter': 'Adapter'}
GROUPS = ['cifar100-r50', 'cifar100-vitb16', 'svhn-r50', 'svhn-vitb16']
DISPLAY = {'cifar100-r50': 'CIFAR-100 / R50', 'cifar100-vitb16': 'CIFAR-100 / ViT',
           'svhn-r50': 'SVHN / R50', 'svhn-vitb16': 'SVHN / ViT'}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                             encoding='utf-8', newline='\n')


def main():
    report = read('reports/P2_report.json')
    selected = read('selected_hparams.json')
    proposal = read('reports/P3_proposal.json')
    audit = read('records/p2/delivery_audit.json')
    assert audit['status'] == 'passed' and report['P2_completed_candidates'] == 48
    assert not report['P3_authorized'] and not report['centralized_test_authorized']
    chosen = {c['config']['configuration_id']: c for c in report['candidates']
              if c['config']['run_id'] == selected['configurations'][c['config']['configuration_id']]['selected_from_run_id']}
    assert len(chosen) == 16
    # 重算 P3 的训练/切片/评价/存储四项，防止将历史预算或规划余量误写成实际消耗。
    training, prefix, evaluation, selected_bytes, best_bytes = 0, 0, 0, 0, 0
    for name, c in chosen.items():
        times = c['timings']
        stable = statistics.median(t['fit_seconds'] for t in times if t['epoch'] >= 2)
        fixed = max(0, c['actual_gpu_seconds_parent'] - sum(t['fit_seconds'] for t in times))
        fixed += max(0, times[0]['fit_seconds'] - stable)
        training += 5 * (50 * stable + fixed) / 3600
        prefix += 15 * statistics.median(t['checkpoint_seconds'] for t in times) / 3600
        data = read('records/data_integrity.json')['datasets'][c['config']['dataset']]
        evaluation += 25 * statistics.median(t['validation_seconds'] for t in times) * data['official_test_n'] / data['validation_n'] / 3600
        selected_bytes += sum(c['summary'][role]['size_bytes'] for role in ['best_checkpoint', 'last_checkpoint', 'task_delta'] if role in c['summary'])
        best_bytes += c['summary']['best_checkpoint']['size_bytes']
    full_budget = 1.25 * (training + prefix + evaluation + 2 + .5)
    storage_scenario = 1.25 * (report['storage']['combined_bytes'] + 5 * selected_bytes + 15 * best_bytes)
    for observed, expected in [(training, proposal['P3_base_training_gpu_hours']),
        (prefix, proposal['P3_extra_prefix_save_gpu_hours_upper_scenario']),
        (evaluation, proposal['centralized_evaluation_gpu_hours_upper_scenario']),
        (full_budget, proposal['P3_training_prefix_evaluation_reserves_with_25_percent_margin_gpu_hours']),
        (storage_scenario, proposal['project_environment_full_retention_with_prefix_and_25_percent_margin_bytes'])]:
        assert math.isclose(observed, expected, rel_tol=1e-10)
    recovery = read('records/p2/storage_race_recovery.json')
    for item in recovery['completed_candidates_preserved_without_rerun']:
        directory = ROOT / 'runs' / item['run_id']
        for key, file in [('summary_sha256', 'summary.json'), ('epochs_sha256', 'epochs.jsonl')]:
            assert hashlib.sha256((directory / file).read_bytes()).hexdigest() == item[key]
    save('records/p2/final_review_checks.json', {'status': 'passed', 'P3_proposal_recalculated': True,
        'initial_17_complete_candidate_summary_and_epochs_unchanged': True,
        'P3_training_gpu_hours': training, 'P3_all_in_scenario_with_margin_gpu_hours': full_budget,
        'P3_storage_scenario_gib': storage_scenario / 2**30, 'P3_authorized': False, 'centralized_test_authorized': False})
    attempts = list(report['gpu_ledger']['attempts'].values())
    by_kind = defaultdict(float)
    for attempt in attempts:
        by_kind[attempt['kind']] += attempt['charged_gpu_seconds'] / 3600
    indices = Counter(c['config']['tuning_candidate_index'] for c in chosen.values())
    score_table = [['数据集 / 骨干', 'Scratch', 'Full FT', 'LP', 'Adapter']]
    for group in GROUPS:
        score_table.append([DISPLAY[group]] + [f'{chosen[group+"-"+m]["selection"]["selection_value"]:.2f}' for m in METHODS])
    lr_table = [['数据集 / 骨干', '方法', '所选 LR', '验证均值 (%)', '网格位置']]
    all_lr = []
    efficiency = [['数据集 / 骨干', '方法', '训练参数 / 百万', '完整轮 / 秒', 'GPU 分配峰值 / GiB']]
    for group in GROUPS:
        for method in METHODS:
            name = group + '-' + method
            c = chosen[name]
            cfg = c['config']
            position = {1: '低边界', 2: '中间', 3: '高边界'}[cfg['tuning_candidate_index']]
            lr_table.append([DISPLAY[group], LABELS[method], f'{cfg["learning_rate"]:g}', f'{c["selection"]["selection_value"]:.4f}', position])
            stable = statistics.median(t['fit_seconds'] for t in c['timings'] if t['epoch'] >= 2)
            efficiency.append([DISPLAY[group], LABELS[method], f'{c["summary"]["parameters"]["trainable_params"]/1e6:.3f}',
                               f'{stable:.2f}', f'{c["summary"]["peak_allocated_bytes"]/2**30:.3f}'])
            candidate_values = sorted(selected['configurations'][name]['candidates'], key=lambda v: v['learning_rate'])
            all_lr.append({'configuration_id': name, 'learning_rates': [v['learning_rate'] for v in candidate_values],
                'mean_val_top1_epochs13_15': [v['selection_value'] for v in candidate_values],
                'selected_learning_rate': cfg['learning_rate'], 'selected_run_id': cfg['run_id']})
    for path in ['reports/P2_selected_configuration_summary.csv', 'reports/P2_LR_comparison.csv']:
        with (ROOT / path).open('w', encoding='utf-8-sig', newline='') as file:
            if 'LR_comparison' in path:
                rows = [{'configuration_id': row['configuration_id'], **{f'LR{i+1}': row['learning_rates'][i] for i in range(3)},
                    **{f'val_mean_LR{i+1}': row['mean_val_top1_epochs13_15'][i] for i in range(3)},
                    'selected_learning_rate': row['selected_learning_rate']} for row in all_lr]
            else:
                rows = [{'configuration_id': n, 'learning_rate': c['config']['learning_rate'],
                    'selection_mean_top1': c['selection']['selection_value'],
                    'median_complete_epoch_seconds': statistics.median(t['fit_seconds'] for t in c['timings'] if t['epoch'] >= 2),
                    'trainable_parameters': c['summary']['parameters']['trainable_params'],
                    'peak_allocated_bytes': c['summary']['peak_allocated_bytes']} for n, c in sorted(chosen.items())]
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save('reports/P2_analysis.json', {'selection_rule': 'mean validation Top-1 epochs13-15, exact ties smaller LR',
        'seed': 123, 'test_evaluation_performed': False, 'selected_LR_comparisons': all_lr,
        'selected_grid_position_counts': dict(indices), 'actual_gpu_hours_by_kind': dict(by_kind),
        'candidate_execution_commit_counts': dict(Counter(c['summary']['execution_commit'] for c in report['candidates'])),
        'results_are_descriptive_validation_only': True})
    storage_gib = report['storage']['combined_bytes'] / 2**30
    budget_rows = [['项目', 'GPU-hours 提案'], ['80 个正式 run 基础训练', f'{training:.3f}'],
        ['最多 240 个额外 prefix 保存', f'{prefix:.3f}'], ['最多 400 个评价角色', f'{evaluation:.3f}'],
        ['诊断预留 + 推理基准预留', '2.000 + 0.500'], ['上述合计并加 25%', f'{full_budget:.3f}'],
        ['建议新阶段独立硬上限（待批准）', '160.000']]
    slices = [['组', '0.25B / 秒', '0.5B / 秒', 'B / 秒']]
    for name, value in sorted(proposal['common_time_slices'].items()):
        b = value['B_seconds_proposal']; slices.append([DISPLAY[name], f'{b*.25:.1f}', f'{b*.5:.1f}', f'{b:.1f}'])
    pages = [
      {'title': 'P2 阶段结果汇报', 'subtitle': '2026-10-22 | PI 阶段审查 | P3 与集中测试关闭', 'items': [
       ('p', f'结论：P2 工程验收通过。48/48 个候选各完成 15 个完整 epoch，共 720 轮、39,935,160 个训练样本和 312,480 次 optimizer 更新。16 个配置的 LR 已按原规则冻结；未发现 OOM 或数值失败。'),
       ('p', f'实际总成本 {report["P2_actual_gpu_hours"]:.4f} / 30 GPU-hours，含全部验收、失败前缀、恢复、重跑和退出。项目及环境 {storage_gib:.2f} GiB / 60 GiB，仍有至少 5 GiB 退出余量。'),
       ('h', '验证集表现概览'), ('table', score_table),
       ('p', '表内数值是各配置所选 LR 的第 13–15 轮验证 Top-1 均值 (%)，仅 seed=123。各格独立调参，不能当作测试成绩或五种子正式方法结论。'),
       ('p', 'CIFAR-100：R50 Full FT 为 83.99%，Adapter 为 82.65%；ViT Adapter 为 92.72%，Full FT 为 92.14%。这些是在当前验证集和预算窗口内的描述性差异。'),
       ('p', 'SVHN：R50 Scratch、Adapter、Full FT 分别为 95.37%、95.03%、94.97%；ViT Full FT 与 Adapter 分别为 96.46%、96.31%。LP 验证均值低于可适配主干的方法，冻结特征在这一设置下受限。'),
       ('p', '阶段结论：已具备将所选 LR 传入 P3 草案的工程条件。泛化、稳定性、显著性和方法优劣仍需原定五种子正式实验及封存后的集中测试；本轮未启动这些工作。') ]},
      {'title': '选参结果与解释边界', 'subtitle': '16 个配置；全部三档 LR 均保留', 'items': [
       ('p', '严格使用第 13、14、15 轮验证正确数总和比较；三轮验证样本数相同，等价于 Top-1 均值。精确并列时选择数值更小的 LR。表中 Full FT 列出主干 LR，分类头 LR 为其 10 倍。'),
       ('table', lr_table),
       ('p', f'{indices[1]+indices[3]}/16 个所选 LR 位于预注册网格边界：低边界 {indices[1]} 个，高边界 {indices[3]} 个，中间 {indices[2]} 个。因此本轮证明的是三档内的选择，不证明 LR 全局或局部最优。没有据此扩展网格或追加候选。'),
       ('p', '高 LR 的 ViT Scratch 产生很低的验证平台：CIFAR-100 的 3e-3 档均值 5.65%，SVHN 的 3e-3 档均值 18.92%；所有记录数值仍有限。低准确率与 NaN 是不同事件，不能把本轮零 NaN 写成历史 NaN 根因已解决。'),
       ('p', '部分配置训练损失继续下降而验证损失回升。该现象支持在 P3 检查泛化和时间预算轨迹，不能据单条曲线证明过拟合机制或修改原选参标准。') ]},
      {'title': '效率与参数开销', 'subtitle': '均采用各配置所选 LR 的真实 P2 测量', 'items': [
       ('p', '完整轮时间取第 2–15 轮 train + validation + checkpoint 完整事务耗时中位数，含实际并发环境。GPU 内存为 torch 记录的 allocated 峰值；它与设备总占用、reserved 峰值不同。'),
       ('table', efficiency),
       ('p', 'ViT Adapter 的训练参数约 3.56–3.63 百万，约为对应 Full FT 的 4.1–4.2%；R50 Adapter 约 10.12–10.30 百万，约为对应 Full FT 的 43%。两种 Adapter 的相对开销不能混称为同一参数比例。'),
       ('p', 'R50 Adapter 的 allocated 峰值高于 Full FT，完整轮也略慢；减少可训练参数并未自动减少激活图或插入模块的代价。ViT Adapter 则在本轮同时减少训练参数、轮时间和 allocated 峰值。'),
       ('p', 'LP 的完整轮耗时大约是对应 Full FT 的 36%–39%，但验证表现有明显代价。效率比较使用完整轮中位数，避免把某个候选额外经历的技术中断成本当成方法固有耗时。') ]},
      {'title': '工程验收、事故与可追溯性', 'subtitle': '56 个 attempt 全结算；无 GPU 任务留在运行状态', 'items': [
       ('p', '独立交付复核逐文件验证 2,133 份原始证据，重新核对所有 epoch 事务链、科学配置 hash、配对 sampler、分类头初始化、样本数、尾窗口和 optimizer 更新数；再精确重算 16 组 LR 选择。'),
       ('p', '真实执行身份由 17 个 6c0a215 完成候选与 31 个 d66193b 完成/恢复候选组成。科学实现基线保持 6c0a215；新运维源码、实际 commit 和验收独立绑定，原 17 个候选 summary/epochs SHA 保持不变。'),
       ('p', '初版 092bcfb 发现终轮摘要恢复和 workers 终止分类问题，两个未完成前缀归档后重新初始化。随后存储统计在原子重命名时发生 FileNotFoundError，17 个完成项保留、两个中断项从第 6/3 轮提交恢复。两次事故均保留现场及费用，没有为改善成绩重跑。'),
       ('p', '最终 18 项 CPU 检查通过，含全部事务边界、身份不匹配、精确选参、共享预算、进程死亡和存储竞态。真实 GPU 恢复覆盖两个正式中断前缀及 R50/ViT 合成检查：恢复状态一致、重复继续更新逐位一致、错误 LR 被拒绝。'),
       ('table', [['资源 / 证据', '最终实测'], ['P2 全部实际 GPU-hours', f'{report["P2_actual_gpu_hours"]:.6f} / 30'],
          ['GPU 验收占用（包含在总数中）', f'{by_kind["gpu_acceptance"]:.6f} GPU-hours'],
          ['项目 + 环境逻辑占用', f'{storage_gib:.3f} GiB / 60 GiB'],
          ['P1 checkpoint 留存及重核', '40 份'], ['P2 checkpoint 留存及重核', '128 份：常规 120 + 技术归档 8'],
          ['模型 checkpoint 下载', '0 份；全部留服务器']]),
       ('p', '数据边界：CIFAR-100 既有完整性记录发现 26 个重复组，其中 14 个跨已分配 split；SVHN 有 1 个重复组但无跨已分配 split 重复。沿用冻结 split，不在调参过程中改数据。相关局限仍需保留。'),
       ('p', '所有验证结果来自单个调参种子；每个数据集使用固定的验证划分，未估计置信区间或做显著性检验。R50 与 ViT 使用不同预训练来源（IN1K / IN21K），跨骨干差异不能单独归因为模型架构。') ]},
      {'title': 'P3 资源提案与进入条件', 'subtitle': '草案待 PI 审核；本次没有 P3 或测试执行许可', 'items': [
       ('p', '原定 16 配置 × 五种子（17、42、2026、3407、65537）= 80 个正式 run，全部从原始初始化训练 50 轮；所选 LR 和 selected_hparams SHA 已解析，不续训 P2。'),
       ('table', budget_rows),
       ('p', f'保存当前全部 P1/P2、P3 best/last/delta 与最多三份额外 prefix，再加 25% 的项目及环境存储情景为 {storage_scenario/2**30:.2f} GiB，建议确认至少 200 GiB。当前 60 GiB 不覆盖该提案；此情景也超过旧 100 GiB 规划线，须 PI 显式批准新的额度或保留政策。'),
       ('p', '算力外推采用所选 P2 候选完整轮时间，已包含实际并发，不重复乘慢化因子。prefix 保存按完整事务费用保守计；最多 400 个评价角色用验证时序和样本数缩放，尚无测试推理。160 GPU-hours 是建议硬上限，须经下一次授权。'),
       ('table', slices),
       ('p', '时间切片建议 B 为每个 dataset/backbone 组四方法 timing-only 的 50 轮估算最小值，取 0.25B、0.5B、B；这些数值尚未冻结，不能按准确率重新选 B。'),
       ('p', '进入 P3 前：批准独立 GPU/存储额度并冻结 B；验收 80-run 入口、不可变 prefix、累计提交时间和 P3 seal；建立唯一 evaluation_id 与按 run+checkpoint SHA 复用预测；在构造 TestData 前严格校验封存、状态、角色和身份。五种子配对分析、Holm 校正及非劣效界限沿用原合同。'),
       ('p', 'P2 工程交付后停止。P3、集中测试和测试预测保持关闭，等待下一次阶段审核。') ]}
    ]
    # 两种交付共享同一内容源，防止 PDF 与 Markdown 的数值或阶段边界不一致。
    markdown = []
    for index, page in enumerate(pages):
        markdown.extend([('# ' if index == 0 else '## ') + page['title'], '', page['subtitle'], ''])
        for kind, content in page['items']:
            if kind == 'table':
                markdown.append('| ' + ' | '.join(content[0]) + ' |')
                markdown.append('|' + '|'.join(['---'] * len(content[0])) + '|')
                markdown.extend('| ' + ' | '.join(row) + ' |' for row in content[1:])
            else:
                markdown.append(('### ' if kind == 'h' else '') + content)
            markdown.append('')
    markdown += ['## 复核与下载', '', '[完整阶段验收回执](P2_验收回执.md) · [16 组曲线](P2_figures/) · '
        '[三档 LR 比较表](P2_LR_comparison.csv) · [48 候选结果](P2_candidate_results.csv) · '
        '[全部 epoch 数据](P2_epoch_curves.csv) · [选参 JSON](../selected_hparams.json) · '
        '[P3 机器可读提案](P3_proposal.json) · [GPU 台账](../records/p2/gpu_ledger.json) · '
        '[模型 SHA 清单](../records/p2/checkpoint_inventory.json) · [独立复核](../records/p2/delivery_audit.json)', '',
        '执行源：`6c0a215` 与 `d66193b`；当前运行验收绑定 `d66193bc3265954094e0badebfcd3279deb69e5a`。归档提交见 Git 历史。', '']
    (ROOT / 'reports/P2_结果汇报.md').write_text('\n'.join(markdown), encoding='utf-8', newline='\n')
    render_pdf(pages)
    print(json.dumps({'report_pages': len(pages), 'selected_configurations': 16, 'P3_budget_proposal_gpu_hours': 160,
                      'P3_storage_proposal_gib': 200, 'P3_authorized': False}))


def render_pdf(pages):
    # 使用嵌入的 TrueType 中文字体；不将字体依赖变成服务器新环境或训练依赖。
    font_path = Path('C:/Windows/Fonts/simhei.ttf')
    assert font_path.exists(), '需要本机可嵌入中文 TrueType 字体'
    pdfmetrics.registerFont(TTFont('CJK', str(font_path)))
    ink = colors.HexColor('#17384B')
    body = ParagraphStyle('Body', fontName='CJK', fontSize=10, leading=15.5, textColor=colors.HexColor('#273642'),
                          spaceAfter=9, wordWrap='CJK', alignment=TA_LEFT)
    title = ParagraphStyle('Title', parent=body, fontSize=21, leading=28, textColor=ink, spaceAfter=7)
    subtitle = ParagraphStyle('Subtitle', parent=body, fontSize=9, leading=14, textColor=colors.HexColor('#64727D'), spaceAfter=15)
    heading = ParagraphStyle('Heading', parent=body, fontSize=12, leading=18, textColor=ink, spaceBefore=5)
    cell = ParagraphStyle('Cell', parent=body, fontSize=8.6, leading=12, spaceAfter=0)
    white = ParagraphStyle('White', parent=cell, textColor=colors.white)
    document = SimpleDocTemplate(str(ROOT / 'reports/P2_阶段汇报.pdf'), pagesize=A4, leftMargin=18*mm,
        rightMargin=18*mm, topMargin=17*mm, bottomMargin=17*mm, title='P2 图像分类迁移学习阶段汇报', author='实验工程记录')
    story = []
    for index, page in enumerate(pages):
        if index: story.append(PageBreak())
        story.extend([Paragraph(escape(page['title']), title), Paragraph(escape(page['subtitle']), subtitle)])
        for kind, content in page['items']:
            if kind != 'table':
                story.append(Paragraph(escape(content), heading if kind == 'h' else body))
                continue
            rows = [[Paragraph(escape(text), white if i == 0 else cell) for text in row] for i, row in enumerate(content)]
            if len(content[0]) == 2: widths = [75*mm, 99*mm]
            elif len(content[0]) == 4: widths = [60*mm, 38*mm, 38*mm, 38*mm]
            elif '所选 LR' in content[0]: widths = [50*mm, 25*mm, 27*mm, 40*mm, 32*mm]
            elif '训练参数 / 百万' in content[0]: widths = [48*mm, 24*mm, 35*mm, 30*mm, 37*mm]
            else: widths = [58*mm, 29*mm, 29*mm, 29*mm, 29*mm]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), ink),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F2F5F7'), colors.white]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6), ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINEBELOW', (0,-1), (-1,-1), .5, colors.HexColor('#D8E0E5'))]))
            story.extend([table, Spacer(1, 10)])
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont('CJK', 8); canvas.setFillColor(colors.HexColor('#64727D'))
        canvas.drawString(18*mm, 9*mm, 'P2 阶段审查 | 验证集单种子调参 | P3 与集中测试关闭')
        canvas.drawRightString(192*mm, 9*mm, str(doc.page)); canvas.restoreState()
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    main()
