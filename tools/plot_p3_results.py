"""P3 验证证据图：五 seed 全保留，不运行测试推理。"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/P3'
METHODS = ['scratch', 'full_ft', 'linear_probe', 'adapter']
LABELS = ['Scratch', 'Full FT', 'Linear probe', 'Adapter']
COLORS = ['#777777', '#356B9A', '#C17C3A', '#408579']
GROUPS = ['cifar100-r50', 'cifar100-vitb16', 'svhn-r50', 'svhn-vitb16']
TITLES = ['CIFAR-100 / R50 IN1K', 'CIFAR-100 / ViT IN21K', 'SVHN / R50 IN1K', 'SVHN / ViT IN21K']
SEEDS = [17, 42, 2026, 3407, 65537]

def load(name):
    with (ROOT / 'reports' / name).open(encoding='utf8') as f:
        return list(csv.DictReader(f))

def matrix(rows, key, xkey, values):
    data = np.array([[float(next(r[key] for r in rows if int(r['seed']) == seed
                and float(r[xkey]) == x)) for x in values] for seed in SEEDS])
    return data.mean(axis=0), data.std(axis=0, ddof=1)

def setup():
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.90, wspace=.32, hspace=.48)
    for i, ax in enumerate(axes.flat):
        ax.set_title(chr(97 + i) + '  ' + TITLES[i], loc='left', pad=11, fontsize=9)
        ax.grid(axis='y', alpha=.13, linewidth=.5)
    return fig, list(axes.flat)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--qa-scripts', type=Path, required=True); args = p.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 9,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 8, 'legend.frameon': False, 'axes.spines.right': False,
        'axes.spines.top': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.linewidth': .7, 'lines.linewidth': 1.2})
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load('P3_validation_runs.csv'); curves = load('P3_epoch_curves.csv'); prefix = load('P3_prefix_validation.csv')

    def save(fig, name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=OUT / (name + '.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(OUT / (name + '.pdf'), facecolor='white')
        fig.savefig(OUT / (name + '.svg'), facecolor='white')
        fig.savefig(OUT / (name + '.png'), dpi=600, facecolor='white')
        plt.close(fig)

    fig, axes = setup()
    for ax, group in zip(axes, GROUPS):
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            vals = np.array([float(r['best_validation_top1']) for r in runs if r['configuration_id'] == group + '-' + method])
            ax.scatter(i + np.linspace(-.12, .12, 5), vals, color=color, s=12, alpha=.8, zorder=3)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt='_', markersize=16,
                color='black', capsize=3, linewidth=.8, zorder=4)
        ax.set_xticks(range(4), LABELS, rotation=18, ha='right', rotation_mode='anchor')
        ax.set_ylabel('Best validation Top-1 (%)'); ax.set_xlim(-.5, 3.5)
    fig.text(.5, .035, 'All five seeds shown; black marks = mean ± sample SD. Validation only.', ha='center', fontsize=8)
    save(fig, 'validation_results')

    fig, axes = setup()
    for ax, group in zip(axes, GROUPS):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            rows = [r for r in curves if r['configuration_id'] == group + '-' + method]
            x = np.arange(1, 51); mean, sd = matrix(rows, 'validation_top1', 'epoch', x)
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.15, linewidth=0)
        ax.set_xlim(1, 50); ax.set_ylim(0, 100)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Top-1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, .997))
    fig.text(.5, .025, 'Mean ± sample SD across five seeds; no smoothing. Validation only.', ha='center', fontsize=8)
    save(fig, 'validation_curves')

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
    fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.80, wspace=.40)
    comparisons = [('cifar100-vitb16', 'linear_probe', 'LP − Full FT'), ('svhn-r50', 'adapter', 'Adapter − Full FT')]
    paired_rows = []
    for i, (ax, (group, method, contrast)) in enumerate(zip(axes, comparisons)):
        differences = []
        for seed in SEEDS:
            own = next(float(r['best_validation_top1']) for r in runs if r['configuration_id'] == group + '-' + method and int(r['seed']) == seed)
            base = next(float(r['best_validation_top1']) for r in runs if r['configuration_id'] == group + '-full_ft' and int(r['seed']) == seed)
            differences.append(own - base)
            paired_rows.append({'comparison': group + '-' + method + '-minus-full_ft', 'seed': seed, 'validation_difference_pp': own - base})
        values = np.array(differences)
        ax.axvline(0, color='#AAAAAA', linewidth=.8, linestyle='--')
        ax.scatter(values, np.arange(5), color=COLORS[2 if i == 0 else 3], s=24, zorder=3)
        ax.errorbar(values.mean(), 5.3, xerr=values.std(ddof=1), color='black', fmt='D', markersize=4, capsize=3)
        ax.set_yticks([0, 1, 2, 3, 4, 5.3], [str(s) for s in SEEDS] + ['Mean ± SD'])
        ax.set_ylim(5.9, -.6); ax.set_ylabel('Paired seed'); ax.set_xlabel(contrast + ' (percentage points)')
        ax.set_title(chr(97 + i) + '  ' + ('CIFAR-100 / ViT IN21K' if i == 0 else 'SVHN / R50 IN1K'), loc='left', pad=12)
    fig.text(.5, .045, 'Best-validation differences are descriptive; official test and confirmatory inference remain closed.', ha='center', fontsize=8)
    save(fig, 'primary_validation_differences')
    with (ROOT / 'reports/P3_primary_validation_differences.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0])); w.writeheader(); w.writerows(paired_rows)

    fig, axes = setup()
    for ax, group in zip(axes, GROUPS):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            rows = [r for r in prefix if r['configuration_id'] == group + '-' + method]
            mean, sd = matrix(rows, 'validation_top1', 'budget_fraction', [.25, .5, 1.])
            ax.errorbar([.25, .5, 1.], mean, yerr=sd, color=color, label=label, marker='o', markersize=3, capsize=2)
        ax.set_xticks([.25, .5, 1.], ['0.25 B', '0.5 B', '1 B'])
        ax.set_xlabel('Frozen own-GPU time budget'); ax.set_ylabel('Validation Top-1 (%)'); ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, .997))
    fig.text(.5, .025, 'Last legal checkpoint before each deadline; mean ± sample SD, n = 5; no retraining.', ha='center', fontsize=8)
    save(fig, 'time_prefix_validation')
    print(json.dumps({'figures': 4, 'backend': 'matplotlib', 'test_data_used': False}))

if __name__ == '__main__':
    main()
