"""独立终态审计和封存；只处理训练/验证证据，绝不调用集中测试。"""
from collections import Counter, defaultdict
import csv
import datetime
import io
import math
import statistics
from tlcls.p3_core import (ROOT, STAGE, candidates, code_commit, execution_identity,
    file_hash, read, require_authorization, scientific_identity, write, atomic_bytes, compatible_execution, execution_identity_at)
from tlcls.p3_budget import Budget, Storage
from tlcls.p3_transaction import EpochTransaction
from tlcls.p3_evaluation import cache_key, evaluation_id

TERMINAL = {'completed', 'numerical_failure', 'oom', 'stalled', 'invalid_protocol'}


def main():
    auth = require_authorization()
    runs = candidates()
    ledger = read(STAGE / 'gpu_ledger.json')
    assert all(a['status'] == 'finalized' for a in ledger['attempts'].values())
    used = Budget.used(ledger)
    assert used <= auth['hard_cap_gpu_seconds']
    summaries, inventory, prefixes, evaluations, table = {}, [], [], [], []
    all_terminal = True
    for config in runs:
        run_id = config['run_id']; directory = ROOT / 'runs' / run_id
        summary = read(directory / 'summary.json') if (directory / 'summary.json').exists() else {'status': 'planned', 'completed_epochs': 0}
        summaries[run_id] = summary
        all_terminal &= summary['status'] in TERMINAL
        identity = scientific_identity(config)
        if summary.get('completed_epochs', 0):
            assert summary['scientific_config_sha256'] == identity['scientific_config_sha256']
            compatible_execution(summary['execution_commit'], summary['implementation'])
            tx = EpochTransaction(directory, identity['scientific_config_sha256'], config)
            current, rows = tx.load(rebuild=False)
            assert len(rows) == summary['completed_epochs']
            count = 45000 if config['dataset'] == 'cifar100' else 65931
            val_count = 5000 if config['dataset'] == 'cifar100' else 7326
            for e, row in enumerate(rows, 1):
                assert row['epoch'] == e and row['train']['samples'] == count
                assert row['validation']['samples'] == val_count
                assert row['cumulative_optimizer_updates'] == e * math.ceil(count / 128)
                assert row['scientific_config_sha256'] == identity['scientific_config_sha256']
                runtime = row['runtime_execution_commit']
                assert row['runtime_execution_identity_sha256'] == execution_identity_at(runtime)['scope_sha256']
                compatible_execution(runtime, execution_identity_at(runtime))
            if summary['status'] == 'completed':
                assert len(rows) == 50
                best = max(rows, key=lambda x: (x['validation']['top1_percent'], -x['epoch']))
                assert current['best_epoch'] == best['epoch'] == summary['selected_epoch']
            roles = [('primary_best_val', current['best']), ('secondary_last', current['last'])]
            for item in tx.prefixes()['items']:
                entry = dict(item, run_id=run_id)
                if item['checkpoint']:
                    assert item['completion']['transaction_completed_gpu_seconds'] <= item['budget_seconds']
                    assert file_hash(tx.path(item['checkpoint']['path'])) == item['checkpoint']['sha256']
                    roles.append(('budget_prefix', item))
                prefixes.append(entry)
            for role, artifact in roles:
                prefix = role == 'budget_prefix'
                cp = artifact['checkpoint'] if prefix else artifact
                item = {'run_id': run_id, 'evaluation_role': role,
                    'budget_fraction': artifact['budget_fraction'] if prefix else None,
                    'budget_seconds': artifact['budget_seconds'] if prefix else None,
                    'checkpoint': str((directory / cp['path']).relative_to(ROOT)).replace('\\', '/'),
                    'checkpoint_sha256': cp['sha256'],
                    'preprocessing_identity': file_hash(ROOT / 'src/tlcls/data.py') + ':' + config['dataset'] + ':' + config['model_key'],
                    'precision_identity': 'BF16_autocast_FP32_logits_loss_TF32_OFF',
                    'prefix_completion': artifact.get('completion') if prefix else None}
                item['evaluation_id'] = evaluation_id(item); item['inference_cache_key'] = cache_key(item)
                evaluations.append(item)
        else:
            for fraction in [.25, .5, 1.]:
                prefixes.append({'run_id': run_id, 'budget_fraction': fraction,
                    'status': 'unavailable' if summary['status'] in TERMINAL else 'pending', 'checkpoint': None})
        assert summary.get('test_evaluator_constructed', False) is False and summary.get('test_metrics') is None
        for path in sorted(directory.glob('checkpoints/*.pt')):
            inventory.append({'run_id': run_id, 'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                              'sha256': file_hash(path), 'size_bytes': path.stat().st_size})
        table.append({'run_id': run_id, 'configuration_id': config['configuration_id'], 'seed': config['seed'],
            'status': summary['status'], 'completed_epochs': summary.get('completed_epochs', 0),
            'selected_epoch': summary.get('selected_epoch'), 'best_validation_top1_percent': summary.get('best_val_top1_percent'),
            'GPU_hours': sum(a['charged_gpu_seconds'] for a in ledger['attempts'].values() if a['run_id'] == run_id) / 3600})
    assert len({x['evaluation_id'] for x in evaluations}) == len(evaluations)
    write(STAGE / 'prefix_manifest.json', {'schema': 'P3_prefix_inventory_v1', 'items': prefixes})
    write(STAGE / 'checkpoint_inventory.json', {'schema': 'P3_checkpoint_inventory_v1', 'items': inventory,
        'total_bytes': sum(x['size_bytes'] for x in inventory), 'P1_P2_retained': True})
    manifest_path = STAGE / 'evaluation_manifest.draft.json'
    write(manifest_path, {'schema': 'P3_evaluation_manifest_v1', 'status': 'draft_test_closed',
        'evaluations': evaluations, 'centralized_test_authorized': False,
        'distinct_prediction_cache_keys': len({x['inference_cache_key'] for x in evaluations})})
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    atomic_bytes(ROOT / 'reports/P3_run_results.csv', stream.getvalue().encode('utf-8'))
    storage = Storage(auth['confirmed_project_environment_allocation_bytes']).snapshot()
    write(STAGE / 'storage_latest.json', storage)
    counts = dict(Counter(s['status'] for s in summaries.values()))
    audit = {'status': 'passed' if all_terminal else 'incomplete', 'run_status_counts': counts,
        'registered_runs': 80, 'completed_epochs': sum(x.get('completed_epochs', 0) for x in summaries.values()),
        'GPU_hours': used / 3600, 'cap_GPU_hours': 160, 'checkpoint_count': len(inventory),
        'storage_GiB': storage['combined_bytes'] / 2**30, 'centralized_test_authorized': False,
        'execution_commit': code_commit(), 'execution_identity': execution_identity()}
    write(STAGE / 'terminal_audit.json', audit)
    if all_terminal:
        seal = {'schema': 'P3_seal_v1', 'status': 'sealed', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_ids': [r['run_id'] for r in runs], 'run_status_counts': counts,
            'summary_sha256': {r['run_id']: file_hash(ROOT / 'runs' / r['run_id'] / 'summary.json') for r in runs},
            'evaluation_manifest_sha256': file_hash(manifest_path),
            'selected_hparams_sha256': file_hash(ROOT / 'selected_hparams.json'),
            'run_manifest_sha256': file_hash(STAGE / 'run_manifest.json'),
            'checkpoint_inventory_sha256': file_hash(STAGE / 'checkpoint_inventory.json'),
            'gpu_ledger_sha256': file_hash(STAGE / 'gpu_ledger.json'),
            'execution_commit': code_commit(), 'execution_identity_sha256': execution_identity()['scope_sha256'],
            'centralized_test_authorized': False, 'P3_C_authorized': False}
        seal_path = ROOT / 'records/P3_sealed.json'
        if seal_path.exists():
            seal['utc'] = read(seal_path)['utc']; assert read(seal_path) == seal
        else: write(seal_path, seal)
    report = '# P3 阶段交接报告\n\n'
    report += f"状态：{'80 个运行终态已独立审计并封存' if all_terminal else '未完成，保持原始失败和暂停证据'}。集中测试继续关闭。\n\n"
    report += f"运行状态：{counts}；完整 epochs：{audit['completed_epochs']}；实际 GPU 占用：{used/3600:.6f} / 160 GPU-hours。项目与环境：{audit['storage_GiB']:.3f} GiB；200 GiB 为历史最低保障，按用户 2026-10-26 指令不再作为存储封顶，保留物理空间退出保护。\n\n"
    report += '冻结 16 组 P2 LR，5 seeds 分母保留；每个正式 run 新初始化并使用 50 epoch horizon、5 epoch warmup。主比较及统计计划不变。现有结果仅为训练和验证证据；官方 test、测试图和发表结论均未解锁。\n\n'
    report += '|run|状态|完整轮次|最佳验证轮次|验证 Top-1|GPU-hours|\n|---|---|---:|---:|---:|---:|\n'
    for x in table:
        report += f"|{x['run_id']}|{x['status']}|{x['completed_epochs']}|{x['selected_epoch']}|{x['best_validation_top1_percent']}|{x['GPU_hours']:.4f}|\n"
    report += '\n下一阶段需要用户对封存收据与评价清单另行明确授权。测试预留不得转用于追加实验。\n'
    atomic_bytes(ROOT / 'reports/P3_阶段交接报告.md', report.encode('utf-8'))
    print(audit['status'])


if __name__ == '__main__': main()
