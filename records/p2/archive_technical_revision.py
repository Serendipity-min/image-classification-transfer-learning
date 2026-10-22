"""保留首批技术前缀与旧验收，供修复后重新登记；不修改 P1，也不删除旧模型。"""

import json
from pathlib import Path
import shutil
import sys


ROOT = Path('/root/user_data/tl_cls_v1')
sys.path.insert(0, str(ROOT / 'src'))
from tlcls.p2_core import STAGE, code_commit, file_hash, process_alive, read, write
from tlcls.p2_budget import Budget


if __name__ == '__main__':
    old = '092bcfb22f18b8f9d07e8964aa24ded3dc2885fb'
    assert code_commit() == old
    owner = read(STAGE / 'scheduler.json')
    assert not process_alive(owner['pid'], owner['start_ticks'])
    budget = Budget()
    assert all(a['status'] == 'finalized' for a in budget.snapshot()['attempts'].values())
    archive = ROOT / 'technical_revisions' / old
    archive.mkdir(parents=True, exist_ok=True)
    records = []
    ids = ['P2-cifar100-r50-full_ft-lr2-s123', 'P2-svhn-vitb16-adapter-lr2-s123']
    for run_id in ids:
        source, target = ROOT / 'runs' / run_id, archive / run_id
        assert source.resolve().is_relative_to(ROOT) and target.resolve().is_relative_to(ROOT)
        assert source.exists() and not target.exists()
        summary = read(source / 'summary.json')
        assert summary['phase'] == 'P2' and summary['completed_epochs'] < 15
        records.append({'run_id': run_id, 'summary_before_archive': summary,
            'scientific_sha256': read(source / 'scientific.json')['scientific_config_sha256'],
            'archived_directory': str(target.relative_to(ROOT)), 'selection_eligible': False})
        source.rename(target)
    history = STAGE / 'acceptance_history' / old
    history.mkdir(parents=True, exist_ok=True)
    for name in ['cpu_acceptance.json', 'gpu_acceptance.json', 'acceptance.json']:
        shutil.copyfile(STAGE / name, history / name)
    for name in ['candidate_plan.json', 'cost_calibration.json', 'forecast.json', 'progress.json', 'stage_status.json']:
        path = STAGE / name
        if path.exists():
            path.rename(history / name)
    if (ROOT / 'selected_hparams.json').exists():
        (ROOT / 'selected_hparams.json').rename(history / 'selected_hparams_incomplete.json')
    for name in ['P2_report.json', 'P3_proposal.json', 'P2_candidate_results.csv', 'P2_epoch_curves.csv']:
        path = ROOT / 'reports' / name
        if path.exists():
            path.rename(history / name)
    with budget.edit() as ledger:
        for item in ledger['attempts'].values():
            if item['run_id'] in ids:
                item.update(execution_commit=old, technical_revision=old,
                            archived_run_directory=str((archive / item['run_id']).relative_to(ROOT)),
                            selection_eligible=False)
        ledger['events'].append({'reason': 'registered_metadata_recovery_and_shutdown_fix',
                                'all_previous_cost_retained': True, 'affected_run_ids': ids})
    receipt = {'old_execution_commit': old,
        'reason': 'terminal epoch summary reconciliation and coordinated worker shutdown classification',
        'technical_reexecution_not_score_search': True, 'original_mathematical_pipeline_unchanged': True,
        'affected_incomplete_prefixes': records, 'P1_changed': False,
        'actual_stage_gpu_hours_before_revision': budget.used(budget.snapshot()) / 3600,
        'budget_cap_unchanged_gpu_hours': 30, 'old_acceptance_retained': str(history.relative_to(ROOT))}
    write(STAGE / 'technical_revisions.json', receipt)
    print(json.dumps({'prefixes_archived_without_deletion': len(records),
                      'all_old_cost_gpu_hours': receipt['actual_stage_gpu_hours_before_revision']}))
