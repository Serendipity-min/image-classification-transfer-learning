"""封存身份之外的 P3-C 监督器；唯一测试入口仍调用原严格授权门。"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path('/root/user_data/tl_cls_v1')
sys.path.insert(0, str(ROOT / 'src'))
from p3c_runtime import (CAP, TOTAL_CAP, EXIT_SECONDS, Phase, read, write, sha, canonical,
    lock, storage_guard, ordered, validate_payload, checked_path, verify_output, publish_output, next_item, forecast, verify_chain)

FILES = ['tools/p3c_runtime.py', 'tools/p3c_supervise.py', 'tools/p3c_acceptance.py', 'tools/p3c_analyze.py']

def scope():
    return {name: sha(ROOT / name) for name in FILES}

def signed_scope():
    stage = ROOT / 'records/p3c'
    key = os.environ.get('TLCLS_TEST_APPROVAL_HMAC_KEY', '').encode()
    payload = read(ROOT / 'records/p3/test_authorization.json')
    approved = validate_payload(payload, key, scope(), sha(stage / 'execution_order.json'))
    seal = read(ROOT / 'records/P3_sealed.json')
    assert approved['seal_sha256'] == sha(ROOT / 'records/P3_sealed.json')
    assert approved['evaluation_manifest_sha256'] == seal['evaluation_manifest_sha256']
    assert approved['P3_B_gpu_ledger_sha256'] == seal['gpu_ledger_sha256'] == sha(stage / 'p3b_gpu_ledger.sealed.json')
    cpu = read(stage / 'fault_injection.json')
    assert cpu['status'] == 'passed' and cpu['source_sha256'] == {Path(n).name:h for n,h in scope().items()}
    assert approved['CPU_fault_injection_sha256'] == sha(stage / 'fault_injection.json')
    return key, approved

def validate_checkpoint(payload, config, scientific_sha):
    if payload['config'] != config or payload['scientific_config_sha256'] != scientific_sha:
        raise ValueError('checkpoint_model_or_scientific_identity_mismatch')

def classification_metrics(logits, labels):
    import numpy as np
    import torch
    prediction = logits.argmax(1); classes = logits.shape[1]; n = len(labels)
    confusion = torch.bincount(labels * classes + prediction, minlength=classes**2).reshape(classes, classes).numpy()
    tp = np.diag(confusion).astype(float)
    recalls = np.divide(tp, confusion.sum(1), out=np.zeros(classes), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * tp, confusion.sum(0) + confusion.sum(1), out=np.zeros(classes),
        where=(confusion.sum(0) + confusion.sum(1)) != 0)
    correct = int((prediction == labels).sum())
    return {'n': n, 'correct': correct, 'top1_percent': 100 * correct / n,
        'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recalls.mean()),
        'nll': float(torch.nn.functional.cross_entropy(logits, labels)), 'confusion_matrix': confusion.tolist()}

def validate_worker_lease(lease, device, synthetic, parent_pid):
    assert lease['status'] == 'running' and lease['device'] == device
    assert lease['guardian_pid'] == parent_pid
    assert lease['kind'] == ('p3c_gpu_acceptance' if synthetic else 'p3c_evaluation')

def worker(args):
    stage = ROOT / 'records/p3c'
    key, approved = signed_scope()
    lease = read(ROOT / 'records/p3/gpu_ledger.json')['attempts'][args.attempt_id]
    validate_worker_lease(lease, args.device, args.synthetic, os.getppid())
    from tlcls.p3_evaluation import require_test_authorization
    # 保留原门禁：真实 checkpoint 哈希仅在正式准入时检查，不增加人工全量复核。
    manifest = require_test_authorization(ROOT / 'records/p3/evaluation_manifest.draft.json')
    from tlcls.p3_core import process_alive
    assert process_alive(lease['guardian_pid'], lease['guardian_start_ticks'])
    assert read(ROOT / 'records/p3/gpu_ledger.json')['attempts'][args.attempt_id]['child_pid'] == os.getpid()
    write(stage / ('worker_progress_' + str(args.device) + '.json'), {'unix': time.time(), 'state': 'authorized'})
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tlcls.common import setup, finite_or_raise
    from tlcls.models import build_model
    torch.cuda.set_device(args.device); setup(0)
    device = torch.device('cuda', args.device)
    registered = {r['run_id']: r for r in read(ROOT / 'records/p3/run_manifest.json')['runs']}
    if args.synthetic:
        config = next(c for c in registered.values() if c['model_key'] == ('r50' if args.device == 0 else 'vitb16')
            and c['method'] == ('full_ft' if args.device == 0 else 'adapter'))
        model, _ = build_model(config, device); model.eval()
        images = torch.zeros(8, 3, 224, 224, device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            first = model(images).float(); second = model(images).float()
        finite_or_raise(first, 'synthetic_logits'); assert torch.equal(first, second)
        write(stage / ('gpu_acceptance_' + str(args.device) + '.json'), {
            'status': 'passed', 'device': args.device, 'inputs': 'synthetic_zero_images_8',
            'test_data_constructed': False, 'repeated_logits_equal': True,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'attempt_id': args.attempt_id, 'supervisor_source_sha256': scope()})
        return
    assert read(stage / 'gpu_acceptance.json')['status'] == 'passed'
    from evaluate import TestData
    items = ordered(manifest['evaluations']); manifest_sha = approved['evaluation_manifest_sha256']
    verified = set()
    while True:
        write(stage / ('worker_progress_' + str(args.device) + '.json'), {'unix': time.time(), 'state': 'dispatch'})
        item = next_item(ROOT, items, manifest_sha, args.device, args.attempt_id, verified)
        if item is None:
            return
        if item == 'WAIT':
            time.sleep(.5); continue
        started = time.monotonic(); identity = item['evaluation_id']
        storage_guard(ROOT)
        checkpoint = checked_path(ROOT, item['checkpoint'])
        # 逐模型只读加载前检查实际来源，防止准入之后误替换文件。
        assert sha(checkpoint) == item['checkpoint_sha256']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = registered[item['run_id']]
        validate_checkpoint(payload, config, read(ROOT / 'runs' / item['run_id'] / 'summary.json')['scientific_config_sha256'])
        model, _ = build_model(config, device); model.load_state_dict(payload['model'], strict=True)
        del payload; model.eval(); torch.cuda.reset_peak_memory_stats()
        dataset = TestData(config['dataset'], config['model_key'])
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        all_logits, all_labels, ids = [], [], []
        with torch.no_grad():
            for images, labels, sample_ids in loader:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images.to(device, non_blocking=True)).float()
                finite_or_raise(logits, 'test_logits')
                all_logits.append(logits.cpu()); all_labels.append(labels); ids.extend(sample_ids)
                write(stage / ('worker_progress_' + str(args.device) + '.json'), {
                    'unix': time.time(), 'state': 'inference', 'evaluation_id': identity})
        logits, labels = torch.cat(all_logits), torch.cat(all_labels)
        prediction = logits.argmax(1); n = len(labels)
        assert n == (10000 if config['dataset'] == 'cifar100' else 26032)
        metrics = classification_metrics(logits, labels)
        metrics.update({
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device': args.device, 'GPU_hours_role_transaction': (time.monotonic() - started) / 3600,
            'cost_attribution': 'role transaction; authorization/startup/idle/exit overhead separately charged in attempt ledger'})
        def writer(file):
            np.savez_compressed(file, sample_id=np.asarray(ids), labels=labels.numpy(), prediction=prediction.numpy(), logits=logits.numpy())
        publish_output(ROOT, item, manifest_sha, writer, metrics, args.attempt_id, lambda: time.monotonic() - started)
        verified.add(identity)
        del model, dataset, loader, logits, labels, prediction, all_logits, all_labels
        torch.cuda.empty_cache()

def supervise_child(args, phase, command, root):
    from tlcls.p3_core import process_alive
    from p3_attempt import parent_death_guard, signal_group
    child = None
    phase.register(args.attempt_id, args.device, args.owner_pid, args.owner_ticks,
        'p3c_gpu_acceptance' if args.synthetic else 'p3c_evaluation')
    reason = 'completed'; stopped = False
    def terminate(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    try:
        child = subprocess.Popen(command, cwd=root, start_new_session=True, preexec_fn=parent_death_guard)
        phase.attach(args.attempt_id, child.pid)
        born = time.time()
        while child.poll() is None:
            used, total = phase.used()
            if used >= CAP - EXIT_SECONDS or total >= TOTAL_CAP - EXIT_SECONDS:
                phase.stop('dual_gpu_budget_exit_guard')
            if stopped or not process_alive(args.owner_pid, args.owner_ticks):
                phase.stop('operator_or_owner_exit')
            try:
                storage_guard(root)
            except RuntimeError:
                phase.stop('physical_storage_exit_guard')
            progress = root / 'records/p3c' / ('worker_progress_' + str(args.device) + '.json')
            updated = max(born, read(progress)['unix']) if progress.exists() else born
            if time.time() - updated > 600:
                phase.stop('worker_no_progress_600_seconds')
            state = read(phase.path)
            if state.get('stop_request'):
                reason = state['stop_request']['reason']
                signal_group(child.pid, signal.SIGTERM)
                if time.monotonic() >= state['stop_request']['deadline_monotonic']:
                    signal_group(child.pid, signal.SIGKILL)
            time.sleep(.25)
        child.wait(); signal_group(child.pid, signal.SIGKILL)
        if child.returncode and reason == 'completed':
            reason = 'worker_failed'; phase.stop(reason)
    except BaseException:
        reason = 'guardian_exception'; phase.stop(reason)
        raise
    finally:
        if child is not None:
            signal_group(child.pid, signal.SIGKILL); child.wait()
        phase.finalize(args.attempt_id, child.returncode if child else -1, reason)
    if child.returncode:
        raise RuntimeError('worker_failed_without_output_publication')

def guardian(args):
    key, _ = signed_scope(); phase = Phase(ROOT, key)
    command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'worker', '--device', str(args.device),
        '--attempt-id', args.attempt_id] + (['--synthetic'] if args.synthetic else [])
    supervise_child(args, phase, command, ROOT)

def launch(args):
    from tlcls.p3_core import process_ticks
    stage = ROOT / 'records/p3c'
    with lock(stage / 'supervisor.lock', blocking=False):
        key, approved = signed_scope(); phase = Phase(ROOT, key)
        write(stage / 'independent_authorization_receipt.json', {'status':'signature_and_scope_verified',
            'approval_id':approved['approval_id'],'authorization_sha256':sha(ROOT/'records/p3/test_authorization.json'),
            'CPU_fault_injection_sha256':approved['CPU_fault_injection_sha256'],
            'supervisor_source_sha256':scope(),'execution_order_sha256':approved['execution_order_sha256'],
            'subcap_gpu_seconds':CAP,'PI_independent_issuance_asserted_in_signed_payload':approved['PI_independent_issuance'],
            'approval_mode':approved.get('approval_mode','independent_PI_issuance'),
            'independent_signing_waived_by_PI':approved.get('independent_signing_waived_by_PI',False),
            'approved_unix':approved.get('approved_unix'),
            'human_identity_authentication':'shared-key HMAC verifies integrity; issuance follows the recorded PI approval mode',
            'secret_logged_or_committed':False})
        shared = read(ROOT / 'records/p3/gpu_ledger.json')
        assert all(a['status'] == 'finalized' for a in shared['attempts'].values())
        own = {i for i,a in shared['attempts'].items() if a['kind'].startswith('p3c_')}
        state = read(phase.path)
        verify_chain(ROOT, key)
        assert own == set(state['attempt_ids'])
        if state['stop_request'] is not None:
            assert args.resume and state['stop_request']['reason'] in ['operator_or_owner_exit', 'worker_failed',
                'guardian_failed', 'supervisor_exit', 'guardian_exception'], 'stopped_phase_requires_PI_review'
        if args.resume:
            assert args.synthetic is False
        else:
            assert not read(stage / 'dispatch.json')['claims'], '已有调度记录须显式resume'
        # 不接管其他用户的 GPU 进程。
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        assert not active.strip(), 'foreign_GPU_process_present'
        storage_guard(ROOT)
        if args.synthetic:
            assert not own, 'GPU入口验收不得静默重试'
        else:
            gpu = read(stage / 'gpu_acceptance.json')
            assert gpu['status'] == 'passed' and gpu['authorization_sha256'] == sha(ROOT/'records/p3/test_authorization.json')
            assert all(proof['supervisor_source_sha256'] == scope() for proof in gpu['devices'])
            manifest = read(ROOT / 'records/p3/evaluation_manifest.draft.json')
            for item in manifest['evaluations']:
                verify_output(ROOT, item, approved['evaluation_manifest_sha256'])
            if args.resume:
                old_dispatch = read(stage / 'dispatch.json')
                write(stage / 'resume_history' / (uuid.uuid4().hex + '.json'), old_dispatch)
                write(stage / 'dispatch.json', {'claims': {}, 'next_ordinal': 0,
                    'resume_history_previous_sha256': sha(stage / 'dispatch.json')})
                with lock(stage / 'phase.lock'):
                    state = read(phase.path); prior = state['stop_request']; state['stop_request'] = None
                    phase.event(state, {'type': 'explicit_technical_resume_same_identity', 'prior_stop': prior})
            execution_order = read(stage / 'execution_order.json')
            prediction = forecast(ROOT, manifest['evaluations'], execution_order['estimated_seconds'], phase.used()[0])
            write(stage / 'forecast.json', prediction)
            assert not prediction['forecast_exceeds_subcap'], 'forecast_exceeds_3_GPU_hours'
        write(stage / 'status.json', {'status': 'synthetic_gpu_acceptance' if args.synthetic else 'running',
            'P3_C_authorized': True, 'supervisor_source_sha256': scope(), 'approval_id': approved['approval_id']})
        children = []
        try:
            for device in [0, 1]:
                command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'guardian', '--device', str(device),
                    '--attempt-id', 'p3c-' + uuid.uuid4().hex, '--owner-pid', str(os.getpid()),
                    '--owner-ticks', process_ticks(os.getpid())] + (['--synthetic'] if args.synthetic else [])
                children.append(subprocess.Popen(command, cwd=ROOT))
            last_forecast = 0
            while any(c.poll() is None for c in children):
                if any(c.poll() not in [None, 0] for c in children):
                    phase.stop('guardian_failed')
                if not args.synthetic and time.monotonic() - last_forecast > 10:
                    prediction = forecast(ROOT, manifest['evaluations'], execution_order['estimated_seconds'], phase.used()[0])
                    write(stage / 'forecast.json', prediction); last_forecast = time.monotonic()
                    if prediction['forecast_exceeds_subcap']:
                        phase.stop('forecast_exceeds_3_GPU_hours')
                time.sleep(.5)
            assert all(c.wait() == 0 for c in children)
        except BaseException:
            phase.stop('supervisor_exit')
            for child in children:
                child.wait()
            raise
        shared = read(ROOT / 'records/p3/gpu_ledger.json')
        assert all(a['status'] == 'finalized' for a in shared['attempts'].values())
        used, total = phase.used(); assert used <= CAP and total <= TOTAL_CAP
        verify_chain(ROOT, key)
        if args.synthetic:
            proofs = [read(stage / ('gpu_acceptance_' + str(i) + '.json')) for i in [0, 1]]
            assert all(x['status'] == 'passed' and not x['test_data_constructed'] for x in proofs)
            write(stage / 'gpu_acceptance.json', {'status': 'passed', 'devices': proofs, 'P3C_gpu_seconds_including_acceptance': used,
                'authorization_sha256':sha(ROOT/'records/p3/test_authorization.json')})
            status = 'ready_for_frozen_test_execution'
        else:
            values = [verify_output(ROOT, item, approved['evaluation_manifest_sha256'])
                for item in read(ROOT / 'records/p3/evaluation_manifest.draft.json')['evaluations']]
            assert len(values) == 400 and all(values)
            with (stage / 'evaluation_receipts.jsonl').open('wb') as f:
                for value in values:
                    f.write(canonical(value) + b'\n')
                f.flush(); os.fsync(f.fileno())
            write(stage / 'status.json', {'status':'400_evaluations_completed_pending_independent_audit'})
            subprocess.run([sys.executable, str(ROOT/'tools/p3c_analyze.py')], cwd=ROOT, check=True)
            status = '400_evaluations_completed_and_independently_audited'
        write(stage / 'status.json', {'status': status, 'P3C_gpu_seconds': used, 'P3_total_gpu_seconds': total,
            'new_training_authorized': False, 'awaiting_PI_review': not args.synthetic})

def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['launch', 'guardian', 'worker'], default='launch')
    p.add_argument('--synthetic', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--device', type=int, choices=[0, 1]); p.add_argument('--attempt-id')
    p.add_argument('--owner-pid', type=int); p.add_argument('--owner-ticks'); args = p.parse_args()
    if args.mode == 'launch':
        launch(args)
    elif args.mode == 'guardian':
        guardian(args)
    else:
        worker(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('P3C_stopped:' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
