"""保留全部候选的原始曲线；单种子调参不产生跨方法统计结论。"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    report = json.loads((root / 'reports/P2_report.json').read_text(encoding='utf-8'))
    selection = json.loads((root / 'selected_hparams.json').read_text(encoding='utf-8'))
    assert len(report['candidates']) == 48
    out = root / 'reports/P2_figures'
    out.mkdir(parents=True, exist_ok=True)
    # 图的证据链是更新轨迹、验证损失、验证准确率；选择窗口来自预注册规则。
    contract = {'claim': 'All registered LR trajectories remain visible, with the pre-registered selection window.',
        'archetype': 'quantitative grid', 'backend': 'python', 'dimensions_mm': [183, 72],
        'panel_roles': ['Train loss: update behavior', 'Validation loss: held-out loss behavior',
                        'Validation Top-1: held-out accuracy and LR selection window'],
        'replicate_unit': 'one seed 123 per LR; epochs are correlated observations, not replicates',
        'uncertainty': 'none; no seed CI, no tests, no method ranking',
        'scope': 'P2 15 epochs of a 50-epoch scheduler; five warmup epochs; validation only',
        'selection': 'mean validation Top-1 epochs 13-15; exact ties smaller numeric LR',
        'source_mapping': {'epoch': 'x', 'train.loss': 'panel a', 'validation.loss': 'panel b',
                           'validation.top1_percent': 'panel c', 'learning_rate': 'series'},
        'exclusions': [], 'smoothing': False, 'transformation': 'none',
        'reuse': 'style only; all P2 grouping and eligibility logic reads current raw receipts',
        'target': 'stage review, not a journal submission', 'minimum_font_pt': 7,
        'exports': ['PDF', 'SVG', 'PNG']}
    (out / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n', encoding='utf-8')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.6})
    palette = [('#9AB8C5', 'o', ':'), ('#4C819B', 's', '--'), ('#164A63', '^', '-')]
    method_names = {'scratch': 'Scratch', 'full_ft': 'Full FT', 'linear_probe': 'Linear probe', 'adapter': 'Adapter'}
    all_rows, qa = [], []
    configurations = sorted({c['config']['configuration_id'] for c in report['candidates']})
    for configuration in configurations:
        group = sorted([c for c in report['candidates'] if c['config']['configuration_id'] == configuration],
                       key=lambda c: c['config']['learning_rate'])
        assert len(group) == 3
        name = 'P2_' + configuration.replace('-', '_')
        fig, axes = plt.subplots(1, 3, figsize=(7.2047244094, 2.8346456693))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.66, wspace=0.42)
        for candidate, (color, marker, linestyle) in zip(group, palette):
            config, epochs = candidate['config'], candidate['epochs']
            x = [e['epoch'] for e in epochs]
            assert x == list(range(1, len(epochs) + 1))
            winner = selection['configurations'][configuration]['selected_from_run_id'] == config['run_id']
            label = 'LR ' + format(config['learning_rate'], '.0e')
            if winner:
                label += ' (selected)'
            if candidate['summary']['status'] != 'completed':
                label += ' [' + candidate['summary']['status'] + ']'
            values = [[e['train']['loss'] for e in epochs], [e['validation']['loss'] for e in epochs],
                      [e['validation']['top1_percent'] for e in epochs]]
            assert all(math.isfinite(v) for series in values for v in series)
            for ax, series in zip(axes, values):
                # 每个完成的原始 epoch 都画点；失败后的空白不插值、不延长曲线。
                ax.plot(x, series, label=label, color=color, marker=marker, markersize=2.5,
                        linewidth=1.25 if winner else 0.9, linestyle=linestyle)
            for e in epochs:
                all_rows.append({'run_id': config['run_id'], 'configuration_id': configuration,
                    'learning_rate': config['learning_rate'], 'status': candidate['summary']['status'],
                    'selected': winner, 'epoch': e['epoch'], 'train_loss': e['train']['loss'],
                    'validation_loss': e['validation']['loss'],
                    'validation_top1_percent': e['validation']['top1_percent'],
                    'scientific_config_sha256': e['scientific_config_sha256']})
        for index, (ax, ylabel) in enumerate(zip(axes, ['Train cross-entropy', 'Validation cross-entropy', 'Validation Top-1 (%)'])):
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 5, 10, 15])
            ax.set_xlim(0.7, 15.3)
            ax.set_ylim(bottom=0)
            ax.tick_params(length=2, width=0.6)
            ax.annotate(chr(ord('a') + index), xy=(0, 1), xycoords='axes fraction',
                        xytext=(-22, 9), textcoords='offset points', weight='bold', fontsize=8)
        axes[2].set_ylim(0, 100)
        axes[2].axvspan(12.5, 15.3, color='#E8ECEF', alpha=0.55, zorder=0)
        cfg = group[0]['config']
        dataset = 'CIFAR-100' if cfg['dataset'] == 'cifar100' else 'SVHN'
        model = 'ResNet-50' if cfg['model_key'] == 'r50' else 'ViT-B/16'
        fig.suptitle(f'{dataset} | {model} | {method_names[cfg["method"]]} | P2 seed 123', y=0.98, fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False)
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=out / f'{name}.alignment.json',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
        fig.savefig(out / f'{name}.pdf', facecolor='white')
        fig.savefig(out / f'{name}.svg', facecolor='white')
        fig.savefig(out / f'{name}.png', dpi=600, facecolor='white')
        plt.close(fig)
        for script, extra, suffix in [
            ('audit_pdf_text.py', ['--min-pt', '5', '--json'], 'text-audit.json'),
            ('audit_figure_collisions.py', ['--json-out', str(out / f'{name}.collision-audit.json')], 'collision-audit.txt')]:
            result = subprocess.run([sys.executable, str(args.qa_scripts / script),
                str(out / f'{name}.pdf'), *extra], capture_output=True, text=True)
            (out / f'{name}.{suffix}').write_text(result.stdout + result.stderr, encoding='utf-8')
            qa.append({'figure': name, 'audit': script, 'exit_code': result.returncode})
            assert result.returncode == 0, (name, script, result.stdout)
    assert len(all_rows) == sum(len(c['epochs']) for c in report['candidates'])
    with (out / 'source_data.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (out / 'automated_qa.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figures': len(configurations), 'source_epoch_rows': len(all_rows), 'backend': 'python'}))


if __name__ == '__main__':
    main()
