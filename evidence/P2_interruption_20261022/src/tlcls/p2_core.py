"""P2 的稳定科学身份与持久化原语；本模块不导入 torch，也不访问测试数据。"""

from collections import defaultdict
import hashlib
import errno
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid


ROOT = Path(os.environ.get('TLCLS_ROOT', '/root/user_data/tl_cls_v1')).resolve()
STAGE = ROOT / 'records/p2'
CAP_SECONDS = 30 * 3600
EXIT_RESERVE_SECONDS = 60
PIPELINE = ['src/tlcls/' + n + '.py' for n in ['common', 'data', 'models', 'engine']]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def reject(value):
        raise ValueError('JSON 非有限值: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def fsync_directory(path):
    # Linux 上持久化 rename 的目录项；事务不能只 fsync 文件内容。
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def code_commit():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def execution_identity():
    names = PIPELINE + ['src/evaluate.py'] + sorted(str(p.relative_to(ROOT)).replace('\\', '/') for pattern in
        ['src/p2_*.py', 'src/tlcls/p2_*.py'] for p in ROOT.glob(pattern))
    files = {name: file_hash(ROOT / name) for name in sorted(set(names))}
    return {'files': files, 'scope_sha256': value_hash(files),
            'mathematical_pipeline_sha256': value_hash({n: files[n] for n in PIPELINE})}


def candidates():
    matrix = read(ROOT / 'configs/run_matrix.json')
    runs = [r for r in matrix['planned_runs'] if r['phase'] == 'P2']
    assert len(runs) == len({r['run_id'] for r in runs}) == 48
    groups = defaultdict(list)
    grid = {'scratch': [3e-4, 1e-3, 3e-3], 'full_ft': [1e-5, 3e-5, 1e-4],
            'linear_probe': [1e-3, 3e-3, 1e-2], 'adapter': [1e-4, 3e-4, 1e-3]}
    for r in runs:
        assert r['seed'] == 123 and r['epochs'] == 15 and r['gpu_count'] == 1
        assert r['scheduler_horizon_epochs'] == 50 and r['warmup_epochs'] == 5
        assert (r['micro_batch_size'], r['gradient_accumulation_steps'], r['effective_batch_size']) == (32, 4, 128)
        assert r['autocast_dtype'] == 'bfloat16' and r['test_during_training'] is False
        assert r['train_fraction'] == 1 and r['split_seed'] == 20261020 and not r['formal_run']
        assert r['head_lr_multiplier'] == (10 if r['method'] == 'full_ft' else 1)
        assert r['dataset'] in ['cifar100', 'svhn'] and r['model_key'] in ['r50', 'vitb16']
        assert r['configuration_id'] == f"{r['dataset']}-{r['model_key']}-{r['method']}"
        assert r['learning_rate'] == grid[r['method']][r['tuning_candidate_index'] - 1]
        groups[r['configuration_id']].append(r)
    assert len(groups) == 16
    for group in groups.values():
        assert sorted(r['tuning_candidate_index'] for r in group) == [1, 2, 3]
    return runs


def scientific_identity(config, implementation=None):
    protocol = read(ROOT / 'configs/protocol.json')
    asset = next(a for a in read(ROOT / 'records/assets.json')['assets']
                 if a['key'] == config['model_key'] + '_model.safetensors')
    # 运行设备、剩余预算、启动时间、attempt 不参与科学 hash；LR 和实现身份必须参与。
    fields = ['protocol_id', 'run_id', 'phase', 'configuration_id', 'dataset', 'model_key',
              'model_id', 'method', 'seed', 'epochs', 'learning_rate', 'head_lr_multiplier',
              'tuning_candidate_index', 'train_fraction', 'split_seed', 'micro_batch_size',
              'gradient_accumulation_steps', 'effective_batch_size', 'autocast_dtype',
              'scheduler_horizon_epochs', 'warmup_epochs', 'test_during_training']
    identity = {'configuration': {k: config[k] for k in fields},
        'protocol_sha256': file_hash(ROOT / 'configs/protocol.json'),
        'run_matrix_sha256': file_hash(ROOT / 'configs/run_matrix.json'),
        'split_sha256': file_hash(ROOT / 'records' / (config['dataset'] + '_split.json')),
        'dependency_lock_sha256': file_hash(ROOT / 'records/dependencies.lock'),
        'weight_sha256': asset['sha256'], 'weight_revision': asset['revision'],
        'optimization': protocol['optimization'], 'precision': protocol['precision_and_determinism'],
        'implementation': implementation or execution_identity()}
    return {'scientific_config': identity, 'scientific_config_sha256': value_hash(identity)}


def require_authorization(require_acceptance=True):
    authorization = read(STAGE / 'authorization.json')
    assert authorization['P2_authorized'] and not authorization['P3_authorized']
    assert authorization['centralized_test_authorized'] is False
    assert authorization['hard_cap_gpu_seconds'] == CAP_SECONDS
    assert authorization['confirmed_project_environment_allocation_bytes'] >= 60 * 2**30
    if require_acceptance:
        receipt = read(STAGE / 'acceptance.json')
        assert receipt['status'] == 'passed' and receipt['execution_commit'] == code_commit()
        assert receipt['execution_identity'] == execution_identity()
        assert receipt['authorization_sha256'] == file_hash(STAGE / 'authorization.json')
    return authorization


def process_alive(pid, start_ticks=None):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0] != 'Z' and (start_ticks is None or fields[19] == str(start_ticks))
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]


def monotonic():
    return time.monotonic()


def infrastructure_exception(exc, stop_requested=False):
    # 不能把科学身份断言或非有限值伪装成基础设施中断。
    if isinstance(exc, (AssertionError, FloatingPointError)):
        return False
    if isinstance(exc, OSError) and exc.errno in [errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EINTR]:
        return True
    if isinstance(exc, RuntimeError):
        return stop_requested or 'DataLoader worker' in str(exc) or 'storage_exit_reserve' in str(exc)
    return False
