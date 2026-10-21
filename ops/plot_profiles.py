"""Python/matplotlib 绘制全部 48 个原始 epoch；单种子曲线不估计置信区间。"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METHODS = [('scratch', 'Scratch', '#737373', 'o', ':'),
           ('full_ft', 'Full FT', '#315A7D', 's', '-'),
           ('linear_probe', 'LP', '#9A6B9E', '^', '--'),
           ('adapter', 'Adapter', '#399E91', 'D', '-.')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-scripts', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    report = json.loads((ROOT / 'reports/P0_P1_report.json').read_text(encoding='utf-8'))
    assert report['P1_completed_runs'] == 16
    out = ROOT / 'reports/figures'
    out.mkdir(exist_ok=True)
    # 曲线回答工程验收问题：四种方法在前三轮内的训练、验证轨迹是否有限且可核查。
    contract = {'claim': 'All raw three-epoch train/validation trajectories are shown for engineering review.',
        'archetype': 'quantitative grid', 'backend': 'python', 'dimensions_mm': [183, 72],
        'panel_roles': ['Training loss: update behavior', 'Validation loss: held-out loss behavior',
                        'Validation Top-1: held-out class accuracy'],
        'replicate_unit': 'one profile seed 314159 per configuration; not a multi-seed aggregate',
        'uncertainty': 'none; n=1, no formal inference', 'exclusions': [], 'source_rows': 48,
        'scope': 'P1 warmup epochs 1-3 of a 50-epoch scheduler; no test data and no formal method ranking',
        'target': 'engineering stage receipt, not a journal submission',
        'exports': ['PDF', 'SVG', 'PNG'], 'minimum_font_pt': 7}
    (out / 'figure_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.6})
    all_rows = []
    for dataset in ['cifar100', 'svhn']:
        for model in ['r50', 'vitb16']:
            name = f'P1_{dataset}_{model}'
            # 固定等宽网格使四张图具有相同物理尺寸；面板字母用物理点偏移。
            # 单位为 inch，对应 183 × 72 mm；显式数值也便于图源静态预检。
            fig, axes = plt.subplots(1, 3, figsize=(7.2047244094, 2.8346456693))
            fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.66, wspace=0.42)
            for method, label, color, marker, style in METHODS:
                item = next(x for x in report['P1'] if (x['dataset'], x['model_key'], x['method']) == (dataset, model, method))
                epochs = item['epochs']
                assert [e['epoch'] for e in epochs] == [1, 2, 3]
                values = [[e['train']['loss'] for e in epochs], [e['validation']['loss'] for e in epochs],
                          [e['validation']['top1_percent'] for e in epochs]]
                for ax, series in zip(axes, values):
                    ax.plot([1, 2, 3], series, label=label, color=color, marker=marker,
                            markersize=3, linewidth=1.1, linestyle=style)
                for e in epochs:
                    all_rows.append({'run_id': item['run_id'], 'dataset': dataset, 'model': model,
                        'method': method, 'epoch': e['epoch'], 'train_loss': e['train']['loss'],
                        'validation_loss': e['validation']['loss'],
                        'validation_top1_percent': e['validation']['top1_percent']})
            for index, (ax, ylabel) in enumerate(zip(axes, ['Train cross-entropy', 'Validation cross-entropy', 'Validation Top-1 (%)'])):
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.set_xticks([1, 2, 3])
                ax.set_xlim(0.9, 3.1)
                ax.set_ylim(bottom=0)
                ax.tick_params(length=2, width=0.6)
                ax.annotate(chr(ord('a') + index), xy=(0, 1), xycoords='axes fraction',
                            xytext=(-22, 9), textcoords='offset points', weight='bold', fontsize=8)
            axes[2].set_ylim(0, 100)
            display_dataset = 'CIFAR-100' if dataset == 'cifar100' else 'SVHN'
            display_model = 'ResNet-50 IN1K' if model == 'r50' else 'ViT-B/16 IN21K'
            fig.suptitle(f'{display_dataset} | {display_model} | P1 profile only, seed 314159', y=0.98, fontsize=8)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=out / f'{name}.alignment.json',
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
            # 保留原始折线，不平滑、不插值，也不把 profile 曲线称作正式学习曲线。
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
                assert result.returncode == 0, (name, script, result.stdout)
    assert len(all_rows) == 48
    with (out / 'source_data.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({'figures': 4, 'raw_epoch_rows': len(all_rows), 'backend': 'python'}))


if __name__ == '__main__':
    main()
