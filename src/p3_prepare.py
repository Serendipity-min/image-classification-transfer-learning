"""登记冻结矩阵和用户资源保障；不启动 CUDA 或读取官方测试标签。"""
from collections import Counter, defaultdict
import datetime
import importlib.metadata
import os
import shutil
import subprocess
from tlcls.p3_core import (ARCHIVE, CAP_SECONDS, ROOT, STAGE, PIPELINE, candidates, code_commit,
    execution_identity, file_hash, read, value_hash, write)
from tlcls.p3_budget import Storage


def immutable(path, value):
    if path.exists():
        assert read(path) == value, '已冻结文件不允许覆盖: ' + path.name
    else:
        write(path, value)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    proposal = read(ROOT / 'reports/P3_proposal.json')
    groups = {}
    for key, group in proposal['common_time_slices'].items():
        b = group['B_seconds_proposal']
        groups[key] = {'B_seconds': b, 'fractions': [.25, .5, 1.0],
            'slice_seconds': [b * f for f in [.25, .5, 1.0]],
            'reason': 'minimum of four P2 timing-only 50-epoch extrapolations; contract section 4.2; no accuracy input'}
    frozen = {'schema': 'P3_time_budget_v1', 'source': 'reports/P3_proposal.json',
        'source_sha256': file_hash(ROOT / 'reports/P3_proposal.json'), 'source_commit': ARCHIVE,
        'frozen_utc': now, 'groups': groups,
        'clock': 'own cumulative actual GPU reservation attempts including initialization, train, validation, full durable saves; queue excluded',
        'eligibility': 'completed validation + full hashed durable transaction before deadline; no retroactive snapshots',
        'test_scores_used': False}
    frozen_path = STAGE / 'time_budget_frozen.json'
    if frozen_path.exists():
        frozen['frozen_utc'] = read(frozen_path)['frozen_utc']
    immutable(frozen_path, frozen)
    selected = read(ROOT / 'selected_hparams.json')['configurations']
    assets = {a['key']: a for a in read(ROOT / 'records/assets.json')['assets']}
    runs = []
    for draft in proposal['resolved_runs']:
        r = {k: v for k, v in draft.items() if k not in
             ['status', 'budget_required_at', 'provenance_required_at', 'gpu_hour_budget']}
        weight = assets[r['model_key'] + '_model.safetensors']
        r.update(learning_rate=selected[r['configuration_id']]['learning_rate'],
                 checkpoint_revision=weight['revision'], checkpoint_sha256=weight['sha256'])
        runs.append(r)
    # 按计时和策略均衡，首批覆盖两骨干、两策略和两设备；不读取分数决定顺序。
    first = ['P3-cifar100-r50-full_ft-s17', 'P3-svhn-vitb16-adapter-s17']
    loads, counts, assignments = [0.0, 0.0], defaultdict(Counter), {}
    basis = proposal['selected_timing_basis']
    def duration(r):
        b = basis[r['configuration_id']]
        return b['epoch_seconds'] * 50 + b['startup_terminal_first_epoch_excess_seconds']
    def assign(r, d):
        assignments[r['run_id']] = d
        loads[d] += duration(r); counts[r['method']][d] += 1
    for d, name in enumerate(first):
        assign(next(r for r in runs if r['run_id'] == name), d)
    for r in sorted(runs, key=lambda r: (-duration(r), r['run_id'])):
        if r['run_id'] not in assignments:
            allowed = [d for d in [0, 1] if counts[r['method']][d] < 10]
            assign(r, min(allowed, key=lambda d: (loads[d], d)))
    for r in runs:
        r['physical_gpu_index'] = assignments[r['run_id']]
    manifest = {'schema': 'P3_runs_v1', 'runs': runs, 'first_calibration_runs': first,
        'calibration_epochs': [2, 3], 'planned_device_load_gpu_seconds': loads,
        'method_device_counts': {k: dict(v) for k, v in counts.items()},
        'selection_source_sha256': file_hash(ROOT / 'selected_hparams.json'), 'P2_archive_commit': ARCHIVE}
    immutable(STAGE / 'run_manifest.json', manifest)
    candidates()
    frozen_names = ['configs/protocol.json', 'configs/run_matrix.json', 'records/assets.json',
        'records/dependencies.lock', 'records/cifar100_split.json', 'records/svhn_split.json',
        'selected_hparams.json', 'reports/P3_proposal.json', 'records/p3/run_manifest.json',
        'records/p3/time_budget_frozen.json', 'records/p2/gpu_ledger.json']
    auth = {'schema': 'P3_authorization_v1', 'contract_id': 'TL-CLS-P3-GATE-20261022-v1.0',
        'P3_A_authorized': True, 'P3_B_conditional_on_acceptance': True, 'P3_C_authorized': False,
        'centralized_test_authorized': False, 'hard_cap_gpu_seconds': CAP_SECONDS,
        'confirmed_project_environment_allocation_bytes': 200 * 2**30,
        'allocation_evidence': {'kind': 'explicit_user_contract_resource_confirmation', 'date': '2026-10-22',
            'contract_minimum_gib': 200,
            'user_statement': '执行合同，P3，当前服务器GPU和硬盘存储条件均支持，我授权执行，但是工作路径不变',
            'interpretation': 'manual allocation assurance under the explicitly invoked 200 GiB contract; not inferred from shared filesystem free space'},
        'test_compute_reserve_gpu_seconds': proposal['centralized_evaluation_gpu_hours_upper_scenario'] * 3600,
        'execution_commit': code_commit(), 'execution_identity': execution_identity(),
        'frozen_inputs': {n: file_hash(ROOT / n) for n in frozen_names}}
    auth_path = STAGE / 'authorization.json'
    if auth_path.exists() and read(auth_path) != auth:
        # 仅 P3-A 无正式运行时可更正工程入口；旧 CPU/GPU 回执必须留存，GPU 成本从不重置。
        assert not any((ROOT / 'runs').glob('P3-*/scientific.json')), '已有正式科学身份，需另行审查代码迁移'
        prior = read(auth_path)
        archive = STAGE / 'acceptance_history' / prior['execution_commit']
        archive.mkdir(parents=True, exist_ok=True)
        for name in ['authorization.json', 'entry_preflight.json', 'cpu_acceptance.json',
                     'gpu_acceptance.json', 'gpu_acceptance_0.json', 'gpu_acceptance_1.json',
                     'acceptance.json', 'dispatch_authorization.json']:
            source = STAGE / name
            if source.exists():
                target = archive / name
                if target.exists(): assert target.read_bytes() == source.read_bytes()
                else: shutil.copyfile(source, target)
        write(auth_path, auth)
    else:
        immutable(auth_path, auth)
    storage = Storage(auth['confirmed_project_environment_allocation_bytes'])
    resources = storage.require(4 * 2**30)
    maximum = max(p.stat().st_size for p in ROOT.glob('runs/**/*.pt'))
    assert resources['allocation_remaining_bytes'] >= 10 * 2**30 + 2 * maximum
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).strip()
    assert len(devices) == 2 and not jobs, 'GPU 入口必须空闲，不占用其它任务设备'
    gpu = []
    for line in devices:
        i, uuid, name, driver, total, free = [x.strip() for x in line.split(',')]
        assert int(total) >= 24000 and int(free) >= 22000
        gpu.append({'index': int(i), 'uuid': uuid, 'name': name, 'driver': driver,
                    'total_memory_mib': int(total), 'free_memory_mib': int(free)})
    verified_assets = []
    for asset in assets.values():
        # 对已冻结资源做文件字节校验，不构造 test dataset 或读取任何标签数组。
        path = ROOT / asset['remote']
        assert path.stat().st_size == asset['size_bytes'] and file_hash(path) == asset['sha256']
        verified_assets.append({'key': asset['key'], 'sha256': asset['sha256'], 'size_bytes': asset['size_bytes']})
    for name in PIPELINE:
        import hashlib
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', ARCHIVE + ':' + name])
        assert hashlib.sha256(blob).hexdigest() == file_hash(ROOT / name), '数学管线改变'
    packages = {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'timm', 'huggingface-hub', 'safetensors']}
    assert packages == {'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124', 'timm': '1.0.12',
                        'huggingface-hub': '0.26.2', 'safetensors': '0.4.5'}
    preflight = {'schema': 'P3_preflight_v1', 'status': 'passed', 'utc': now,
        'project_root': str(ROOT), 'environment': str(storage.environment), 'allocation_evidence': auth['allocation_evidence'],
        'actual_allocation_bytes': storage.allocation, 'storage': resources,
        'maximum_existing_checkpoint_bytes': maximum, 'atomic_two_checkpoint_reserve_bytes': 4 * 2**30,
        'filesystem': {'device': ROOT.stat().st_dev, 'free_inodes': os.statvfs(ROOT).f_favail,
            'mount': subprocess.check_output(['findmnt', '-T', str(ROOT), '-n', '-o', 'FSTYPE,TARGET,OPTIONS'], text=True).strip(),
            'quota_tools': {p: bool(shutil.which(p)) for p in ['quota', 'xfs_quota']}},
        'gpu_devices': gpu, 'packages': packages, 'verified_assets': verified_assets,
        'frozen_inputs': auth['frozen_inputs'], 'execution_commit': code_commit(),
        'execution_identity': execution_identity(), 'run_count': len(runs), 'seeds': [17, 42, 2026, 3407, 65537],
        'P2_ledger_sha256_read_only': file_hash(ROOT / 'records/p2/gpu_ledger.json'),
        'test_dataset_constructed': False, 'backup_policy': 'all P1/P2 evidence retained in original paths; Git private archive; checkpoints stay on server'}
    assert preflight['filesystem']['free_inodes'] > 100000
    write(STAGE / 'entry_preflight.json', preflight)
    write(STAGE / 'storage_latest.json', resources)
    print('P3 preparation and real resource preflight passed; P3-B remains gated')


if __name__ == '__main__':
    main()
