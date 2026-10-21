"""由验收证据生成中文阶段回执；所有预算保留估算、预留和未冻结的身份。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    report = read('reports/P0_P1_report.json')
    audit = read('records/acceptance_audit.json')
    addendum = read('reports/acceptance_addendum.json')
    budget = report['next_budget']
    revision = read('records/scheduling_revision.json')
    storage = addendum['storage']
    checkpoint = budget['storage_estimate']
    amount = addendum['budget_proposal']
    full_disk = storage['project_file_bytes'] / 2**30 + 8 * checkpoint['P1_measured_checkpoint_gib']
    slices_disk = full_disk + checkpoint['P3_three_additional_slice_weights_upper_bound_gib']
    descriptions = {'adapter': 'Adapter', 'full_ft': 'Full FT', 'linear_probe': 'LP', 'scratch': 'Scratch'}
    lines = [
        '# P0–P1 阶段验收回执', '',
        '协议：TL-CLS-20261020-v1.0。当前结论：P0 的 16 项配置检查通过；P1 的 16 组全部完成 3 个完整 epoch，共 48 epoch。', '',
        f"GPU 预算台账累计 **{audit['actual_gpu_hours']:.4f} GPU-hours / 4 GPU-hours**，含两轮 P0 实现验收、短程训练、调度修订期间的连续计时及双卡争用基准。早期 BF16 功能探针没有单独精确计时。测试集模型评价次数为 0；P2/P3 未启动。", '',
        '## 环境与代码', '',
        '- 服务器项目：`/root/user_data/tl_cls_v1`；环境：`/root/miniconda3/envs/tl_cls_v1`。',
        '- 两张 RTX 3090，各 24 GiB；cgroup 配额 16 vCPU、64 GiB 内存。共享盘剩余空间不等于租户配额；租户配额仍待确认，按合同 100 GiB 规划。',
        '- Python 3.12.3、torch 2.5.1+cu124、torchvision 0.20.1+cu124；授权后离线安装 timm 1.0.12、huggingface-hub 0.26.2、safetensors 0.4.5。项目 venv 复用基础包，基础环境未修改。',
        '- BF16 两卡功能检查及 torchvision NMS 检查通过；FP32 参数与损失、BF16 autocast，无 GradScaler、TF32、compile 或 LP 特征缓存。',
        f"- 当前代码 commit：`{revision['new_code_commit']}`。前 6 组使用 `{revision['old_code_commit']}`；二者仅调度器文件不同，训练、模型、数据代码逐字相同。", '',
        '调度修订纠正了按矩阵奇偶分配 GPU 导致的方法与设备绑定。活动训练子进程完成后才继续调度，未重启该训练；两版验收记录均保留。最终每种策略各有 2 组在 GPU0、2 组在 GPU1。原始收据保留真实 commit，没有回填新 commit。', '',
        '## 数据、权重与实现验收', '',
        '| 数据 | 训练 | 验证 | 官方测试（仅结构核对） | split SHA-256 |',
        '|---|---:|---:|---:|---|']
    for dataset, item in report['data']['datasets'].items():
        lines.append(f"| {dataset} | {item['train_n']} | {item['validation_n']} | {item['official_test_n']} | `{item['split_sha256']}` |")
    lines += ['', '官方档案 MD5 与本地/服务器 SHA-256 均核对。CIFAR-100 使用细类标签；SVHN 直接使用 torchvision 的 10→0 映射，不额外减一，不使用 extra，不做水平翻转。所有策略输入均为 224，标准化随指定骨干冻结。', '',
        '像素审计：CIFAR-100 共 26 组精确重复，其中 14 组跨分配后的集合；SVHN 共 1 组、无跨集合重复。按合同保存重复清单并保留官方样本，未据此改划分；这限制了对样本完全独立的表述。', '',
        '| 权重 | 仓库 revision | 权重 SHA-256 |', '|---|---|---|']
    for asset in read('records/assets.json')['assets']:
        if asset['key'].endswith('_model.safetensors'):
            model_id = next(r['model_id'] for r in read('configs/run_matrix.json')['planned_runs']
                            if asset['key'] == r['model_key'] + '_model.safetensors')
            lines.append(f"| `{model_id}` | `{asset['revision']}` | `{asset['sha256']}` |")
    lines += ['', 'P0 实测与合同整数一致：LP 仅头更新；Adapter 原主干与原 BN buffers 固定、新分支更新；优化器参数 ID 与白名单精确相等。两种 Adapter 初始化输出最大误差为 0，三次更新后前部 down 投影已有非零有限梯度。', '',
        '累积末窗口 [3,3,1] 与完整 batch 的参数更新最大误差为 0；persistent worker 增强与主进程重建一致。两个骨干恢复前后 eval logits 最大误差为 0，继续更新后的模型与优化器状态逐位一致。实际训练四策略的分类头 hash 及各轮 sampler hash 已再次配对核对。', '',
        '## P1 完成表', '',
        '每组 seed=314159，候选中间学习率，使用正式 50 轮调度的前三轮。下表为验证集最佳 Top-1，只用于运行状态检查；所有轮次仍处于 5 轮 warmup 内，不能据此形成正式方法排名。完整参数、显存、存盘、启动与逐轮日志见 CSV 和逐 run 收据。', '',
        '| 数据 | 骨干 | 策略 | epoch | 最佳 val Top-1 % | 第2/3轮平均秒 | 峰值 allocated / reserved GiB |',
        '|---|---|---|---:|---:|---:|---:|']
    for item in report['P1']:
        s = item['summary']
        lines.append(f"| {item['dataset']} | {item['model_key']} | {descriptions[item['method']]} | 3 | {s['best_val_top1_percent']:.3f} | {item['epoch2_3_mean_fit_seconds']:.2f} | {s['peak_allocated_bytes']/2**30:.2f} / {s['peak_reserved_bytes']/2**30:.2f} |")
    lines += ['', f"合计处理训练样本 {audit['processed_training_samples']:,} 次、optimizer 更新 {audit['optimizer_updates']:,} 次。每轮包含全部训练/验证样本，CIFAR 末有效 batch=72、SVHN=11，未丢尾批。未观察到 OOM 或非有限值。", '',
        '原稿静态排查：提供了文稿，未提供历史可执行代码、日志、checkpoint。因此不能确定旧 NaN 根因，也不能声称精确重现历史失败。当前协议已改动划分、精度、初始化与增强；本轮 3 epoch 无异常不能证明 50 epoch 稳定，也不能归因于某一个改动。', '',
        '## 双卡争用', '',
        '代表配置独占与并发均使用真实训练数据，同样 2 次热身、6 个有效更新、768 个测量样本；完整逐配置三轮时间来自单卡独占运行。并发基准是短程样本，尚不足以覆盖全部方法组合。', '',
        '| 代表配置 | 独占秒 | 并发秒 | 并发/独占 | 数据等待：独占/并发 秒 |', '|---|---:|---:|---:|---:|']
    for item in report['concurrency']:
        lines.append(f"| {item['run_id']} | {item['solo']['seconds']:.3f} | {item['concurrent']['seconds']:.3f} | {item['slowdown_ratio']:.3f} | {item['solo']['data_wait_seconds']:.3f} / {item['concurrent']['data_wait_seconds']:.3f} |")
    lines += ['', '## 后续预算提案', '',
        '由 16 组第 2、3 轮的训练、验证、常规 checkpoint 时间均值外推；另外计入每次启动、首轮额外开销及终结开销。已计入每轮的保存时间不重复加。25% 为计划余量，不是已消耗时间。学校服务器无计价，金额项不适用。', '',
        '| 阶段 | 数量 | 训练 GPU-hours 估计 | 两卡墙钟估计（含实测争用情景，未加余量） |',
        '|---|---|---:|---:|']
    for phase, count in [('P2', '48×15 epoch'), ('P3', '80×50 epoch')]:
        schedule = budget['two_gpu_schedule_proposal'][phase]
        lines.append(f"| {phase} | {count} | {budget[phase + '_gpu_hours']:.2f} | {schedule['two_gpu_wall_hours_with_contention_scenario']:.2f} h |")
    lines += ['', f"训练合计 {amount['P2_P3_training_gpu_hours']:.2f} GPU-hours；训练加 25% 余量 {budget['P2_P3_gpu_hours_with_25_percent_margin']:.2f} GPU-hours；两卡排程加争用情景和 25% 余量约 {budget['two_gpu_wall_hours_with_25_percent_margin']:.2f} 小时。两卡任务分配详见预算 JSON，状态仍为提案。", '',
        f"补充预算：集中评价 80 个 best_val 与 80 个 last，按验证耗时和样本比例估计 {amount['centralized_best_plus_last_test_estimate_gpu_hours']:.2f} GPU-hours；诊断预留上限 2.00 GPU-hours，推理基准预留 0.50 GPU-hours。三项均未执行。训练及这些附加项统一加 25% 后约 **{amount['training_evaluation_diagnostics_inference_with_25_percent_margin_gpu_hours']:.2f} GPU-hours**。", '',
        '预算切片权重与对应评价需在 P2 后冻结时间上限，再单独补计存盘和评价成本；当前总额不含该未冻结扩展。不得把短程估算视作长期资源保证。', '',
        f"当前项目文件约 {storage['project_file_bytes']/2**30:.2f} GiB，其中 P1 checkpoint {checkpoint['P1_measured_checkpoint_gib']:.2f} GiB。若 P1/P2/P3 全部保留 best、last 和任务增量，项目约 {full_disk:.2f} GiB，另加 25% 约 {1.25*full_disk:.2f} GiB。若 80 次正式运行各额外保留 3 份完整切片权重，上界再增加 {checkpoint['P3_three_additional_slice_weights_upper_bound_gib']:.2f} GiB；相应项目加余量约 {1.25*slices_disk:.2f} GiB。", '',
        '长期阶段必须先确认租户配额，并冻结 P2 归档/optimizer 保留政策和 P3 切片数量。当前 checkpoint 留服务器；没有执行任何 checkpoint 删除。', '',
        ('全部保留续跑状态并加余量已超出 100 GiB 规划线。最小处理建议：先确认更大租户额度；若仍限 100 GiB，则在 P2 阶段审查后决定归档/删除哪些 optimizer 状态，并重新核算 P3 切片预算。' if 1.25 * full_disk > 100 else
         '基础完整保留方案尚在 100 GiB 规划线内；加入预算切片后仍需复核，不能按共享盘剩余空间直接扩容。'), '',
        '## 文件与未决事项', '',
        '- 结构化回执：`P0_P1_report.json`；补充计价、存储与评价预算：`acceptance_addendum.json`。',
        '- 完成表：`P1_acceptance_table.csv`；原始逐轮：`../runs/<run_id>/epochs.jsonl`；图源数据及曲线：`figures/`。',
        '- 验收审计：`../records/acceptance_audit.json`；回收文件 hash 清单：`../records/evidence_manifest.json`；40 份 checkpoint 核验索引：`../records/checkpoint_inventory.json`。',
        '- 数据与权重上传约 0.93 GB，低于用户规定的 10 GB 待上传量门槛；所有文件也低于原 1 GB 单文件门槛。三份离线依赖约 3.23 MB。经本地暂存、上传、服务器 SHA 核验后清理成功暂存。',
        '- 慢源产生的旧 CIFAR 未完成下载约 54.5 MB 未上传、未通过完整性校验，按失败文件保留规则仍留在本机；不用于任何实验。',
        '- 当前阶段的历史安装授权等待回执保留为审计历史，最终状态以本回执与完成日志为准。', '',
        '待阶段评审：确认后续算力预算、租户存储配额和归档政策；若争用超过 10%，需完成瓶颈调查后冻结并发排程。P2/P3 需要后续授权，本轮未启动。', '']
    (ROOT / 'reports/P0_P1_验收回执.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'receipt': str(ROOT / 'reports/P0_P1_验收回执.md'),
                      'full_retention_project_with_margin_gib': 1.25 * full_disk,
                      'with_three_slice_weights_and_margin_gib': 1.25 * slices_disk}, ensure_ascii=False))


if __name__ == '__main__':
    main()
