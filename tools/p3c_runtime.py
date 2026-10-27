"""P3-C 独立运维原语；不改封存源码，不访问数据集。"""
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import time
import uuid

CAP = 10800
TOTAL_CAP = 576000
EXIT_SECONDS = 60
ROLES = ['primary_best_val', 'secondary_last', 'budget_prefix']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def write(path, value):
    atomic(path, canonical(value) + b'\n')

@contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def storage_guard(root, free=None):
    remaining = shutil.disk_usage(root).free if free is None else free
    if remaining < 14 * 2**30:
        raise RuntimeError('physical_storage_below_10_plus_4_GiB')
    return remaining

def ordered(items):
    ids = [x['evaluation_id'] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_evaluation_id')
    return sorted(items, key=lambda x: (ROLES.index(x['evaluation_role']), x['run_id'], x.get('budget_fraction') or 0))

def checked_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('path_outside_project')
    return path

def validate_payload(payload, key, source_files, order_sha):
    if len(key) < 32:
        raise PermissionError('external_PI_key_required')
    signed = dict(payload); signature = signed.pop('signature_hmac_sha256', '')
    if not hmac.compare_digest(signature, hmac.new(key, canonical(signed), hashlib.sha256).hexdigest()):
        raise PermissionError('invalid_signature')
    if not (signed['schema'] == 'P3_C_authorization_v1' and signed['centralized_test_authorized'] is True
            and signed['separate_user_approval'] is True and signed['PI_independent_issuance'] is True
            and signed['subcap_gpu_seconds'] == CAP and signed['total_cap_gpu_seconds'] == TOTAL_CAP
            and signed['approval_id'] and signed['execution_order_sha256'] == order_sha
            and signed['supervisor_source_sha256'] == source_files):
        raise PermissionError('authorization_scope_mismatch')
    return signed

def verify_output(root, item, manifest_sha):
    directory = Path(root) / 'records/p3c/results' / item['evaluation_id']
    receipt = directory / 'completed.json'
    if not receipt.exists():
        return None
    value = read(receipt)
    for k in ['evaluation_id', 'run_id', 'evaluation_role', 'checkpoint_sha256']:
        if value[k] != item[k]:
            raise ValueError('completed_output_wrong_identity')
    if value['manifest_sha256'] != manifest_sha or value['status'] != 'completed':
        raise ValueError('completed_output_wrong_manifest')
    prediction = checked_path(root, value['predictions_path'])
    if not prediction.is_relative_to(directory.resolve()) or sha(prediction) != value['predictions_sha256']:
        raise ValueError('completed_prediction_hash_mismatch')
    return value

def publish_output(root, item, manifest_sha, writer, metrics, attempt_id, elapsed, fault=None):
    fault = fault or (lambda _: None)
    storage_guard(root)
    directory = Path(root) / 'records/p3c/results' / item['evaluation_id']; directory.mkdir(parents=True, exist_ok=True)
    target = directory / (attempt_id + '.' + uuid.uuid4().hex + '.npz')
    tmp = target.with_suffix('.npz.tmp')
    with tmp.open('xb') as f:
        writer(f); f.flush(); os.fsync(f.fileno())
    fault('after_file_sync')
    digest = sha(tmp); os.replace(tmp, target)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    fault('after_prediction_publish')
    value = {k: item.get(k) for k in ['evaluation_id', 'run_id', 'evaluation_role', 'budget_fraction',
        'budget_seconds', 'checkpoint_sha256', 'inference_cache_key']}
    duration = elapsed()
    metrics = dict(metrics, GPU_hours_role_transaction=duration / 3600)
    value.update(status='completed', predictions_path=target.relative_to(Path(root)).as_posix(),
        predictions_sha256=digest, manifest_sha256=manifest_sha, metrics=metrics,
        attempt_id=attempt_id, evaluation_wall_seconds_in_attempt=duration,
        test_used_for_selection=False, completed_unix=time.time())
    write(directory / 'completed.json', value)
    fault('after_receipt_publish')
    return value

class Phase:
    def __init__(self, root, key):
        self.root = Path(root); self.stage = self.root / 'records/p3c'
        self.key = key
        self.path = self.stage / 'evaluation_ledger.json'

    def _shared(self):
        from tlcls.p3_budget import Budget
        return Budget(self.root / 'records/p3/gpu_ledger.json', TOTAL_CAP)

    def used(self):
        from tlcls.p3_budget import Budget
        shared = read(self.root / 'records/p3/gpu_ledger.json')
        own = {i: a for i, a in shared['attempts'].items() if a['kind'].startswith('p3c_')}
        return Budget.used({'attempts': own}), Budget.used(shared)

    def event(self, value, event):
        previous = value.get('chain_tip_sha256')
        record = dict(event, previous_sha256=previous, utc_unix=time.time(),
            P3_B_baseline_sha256=value['P3_B_gpu_ledger_sha256'],
            shared_gpu_ledger_sha256=sha(self.root / 'records/p3/gpu_ledger.json'))
        record['signature_hmac_sha256'] = hmac.new(self.key, canonical(record), hashlib.sha256).hexdigest()
        digest = hashlib.sha256(canonical(record)).hexdigest()
        path = self.stage / 'ledger_chain' / (digest + '.json')
        write(path, record)
        value['chain_tip_sha256'] = digest
        write(self.path, value)

    def register(self, attempt_id, device, owner_pid, owner_ticks, kind):
        from tlcls.p3_core import process_ticks
        with lock(self.stage / 'phase.lock'):
            value = read(self.path)
            baseline = read(self.stage / 'p3b_gpu_ledger.sealed.json')
            shared = read(self.root / 'records/p3/gpu_ledger.json')
            assert all(shared['attempts'][i] == a for i, a in baseline['attempts'].items()), 'P3_B_charges_changed'
            phase_used, total = self.used()
            assert phase_used + EXIT_SECONDS < value['subcap_gpu_seconds'] == CAP
            assert total + EXIT_SECONDS < TOTAL_CAP and value.get('stop_request') is None
            assert all(a['status'] == 'finalized' or a['device'] != device for a in shared['attempts'].values())
            self._shared().register(attempt_id, 'P3C-' + attempt_id, device, kind, owner_pid, owner_ticks)
            value['attempt_ids'].append(attempt_id)
            self.event(value, {'type': 'attempt_registered', 'attempt_id': attempt_id, 'device': device,
                'guardian_pid': os.getpid(), 'guardian_start_ticks': process_ticks(os.getpid())})

    def finalize(self, attempt_id, returncode, reason):
        with lock(self.stage / 'phase.lock'):
            value = read(self.path)
            shared = read(self.root / 'records/p3/gpu_ledger.json')
            if shared['attempts'][attempt_id]['status'] == 'finalized':
                return
            self._shared().finalize(attempt_id, returncode, reason, time.monotonic())
            self.event(value, {'type': 'attempt_finalized', 'attempt_id': attempt_id,
                'returncode': returncode, 'reason': reason})

    def attach(self, attempt_id, pid):
        with lock(self.stage / 'phase.lock'):
            value = read(self.path)
            self._shared().update(attempt_id, child_pid=pid, child_pgid=pid)
            self.event(value, {'type': 'child_attached', 'attempt_id': attempt_id, 'pid': pid})

    def stop(self, reason):
        with lock(self.stage / 'phase.lock'):
            value = read(self.path)
            if value.get('stop_request') is None:
                value['stop_request'] = {'reason': reason, 'deadline_monotonic': time.monotonic() + 10}
                self.event(value, {'type': 'stop', 'reason': reason})

def verify_chain(root, key):
    stage = Path(root) / 'records/p3c'; value = read(stage / 'evaluation_ledger.json')
    pointer = value.get('chain_tip_sha256'); registered = set(); visited = set()
    while pointer:
        if pointer in visited:
            raise ValueError('ledger_chain_cycle')
        visited.add(pointer)
        record = read(stage / 'ledger_chain' / (pointer + '.json'))
        assert hashlib.sha256(canonical(record)).hexdigest() == pointer
        signed = dict(record); signature = signed.pop('signature_hmac_sha256')
        assert hmac.compare_digest(signature, hmac.new(key, canonical(signed), hashlib.sha256).hexdigest())
        assert record['P3_B_baseline_sha256'] == value['P3_B_gpu_ledger_sha256']
        if record['type'] == 'attempt_registered':
            registered.add(record['attempt_id'])
        pointer = record['previous_sha256']
    assert registered == set(value['attempt_ids'])
    return len(visited)

def next_item(root, items, manifest_sha, device, attempt_id, verified_ids=None):
    stage = Path(root) / 'records/p3c'
    verified_ids = verified_ids if verified_ids is not None else set()
    with lock(stage / 'dispatch.lock'):
        state = read(stage / 'dispatch.json')
        for role in ROLES:
            group = [x for x in items if x['evaluation_role'] == role]
            pending = []
            for item in group:
                identity = item['evaluation_id']
                if identity in verified_ids:
                    continue
                if verify_output(root, item, manifest_sha):
                    verified_ids.add(identity)
                else:
                    pending.append(item)
            if not pending:
                continue
            for item in pending:
                if item['evaluation_id'] not in state['claims']:
                    state['claims'][item['evaluation_id']] = {'device': device, 'attempt_id': attempt_id,
                        'claimed_unix': time.time(), 'ordinal': state['next_ordinal']}
                    state['next_ordinal'] += 1
                    write(stage / 'dispatch.json', state)
                    return item
            return 'WAIT'
        return None

def forecast(root, items, estimates, used):
    observed = {}
    pending = []
    for item in items:
        path = Path(root) / 'records/p3c/results' / item['evaluation_id'] / 'completed.json'
        if path.exists():
            receipt = read(path)
            key = item['run_id'].rsplit('-s', 1)[0]
            observed.setdefault(key, []).append(receipt['evaluation_wall_seconds_in_attempt'])
        else:
            pending.append(item)
    import statistics
    remaining = 0
    for item in pending:
        key = item['run_id'].rsplit('-s', 1)[0]
        estimate = estimates[item['evaluation_id']]
        if key in observed:
            estimate = max(estimate, statistics.median(observed[key]) * 1.25)
        remaining += estimate
    return {'actual_P3C_gpu_seconds_including_active': used, 'pending': len(pending),
        'remaining_seconds_with_margin': remaining, 'forecast_gpu_seconds': used + remaining + EXIT_SECONDS,
        'subcap_gpu_seconds': CAP, 'forecast_exceeds_subcap': used + remaining + EXIT_SECONDS > CAP,
        'running_partial_work': 'pending role conservatively counted in full; no accuracy used'}
