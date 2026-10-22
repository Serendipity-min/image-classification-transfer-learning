"""将新的事务保存/恢复路径绑定到真实 GPU 模型；所有占用记入 P2 台账。"""

import copy
import math
import os
from pathlib import Path
import tempfile

import torch

from tlcls.common import rng_state, setup, tensor_hash
from tlcls.engine import UpdateScheduler, effective_update
from tlcls.models import build_model, make_optimizer
from tlcls.p2_budget import Budget
from tlcls.p2_core import (ROOT, STAGE, candidates, code_commit, execution_identity, file_hash,
                           read, require_authorization, scientific_identity, write)
from tlcls.p2_transaction import EpochTransaction
from p2_train import restore_checkpoint


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def same(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    require_authorization(require_acceptance=False)
    prior = read(STAGE / 'cpu_acceptance.json')
    assert prior['status'] == 'passed' and prior['execution_commit'] == code_commit()
    assert prior['execution_identity'] == execution_identity()
    attempt = Budget().snapshot()['attempts'][os.environ['TLCLS_ATTEMPT_ID']]
    assert attempt['kind'] == 'gpu_acceptance' and attempt['device'] == 0
    torch.cuda.set_device(0)
    setup(123)
    device = torch.device('cuda', 0)
    generator = torch.Generator().manual_seed(12345)
    batches = [(torch.randn(2, 3, 224, 224, generator=generator), torch.tensor([i, i + 1]),
                [f'P2_acceptance/synthetic/{2*i}', f'P2_acceptance/synthetic/{2*i+1}']) for i in range(3)]
    results = {}
    # 真实中断前缀只读恢复，验证新运维代码可继续原状态；不把诊断更新写回正式 run。
    for run_id in ['P2-svhn-r50-scratch-lr1-s123', 'P2-svhn-r50-full_ft-lr1-s123']:
        config = next(c for c in candidates() if c['run_id'] == run_id)
        identity = scientific_identity(config)
        tx = EpochTransaction(ROOT / 'runs' / run_id, identity['scientific_config_sha256'])
        commit, rows = tx.load(rebuild=False)
        payload = torch.load(tx.path(commit['last']['path']), map_location='cpu', weights_only=False)
        hashes, optimizers = [], []
        for _ in range(2):
            restored, _ = build_model(config, device)
            optimizer, _ = make_optimizer(restored, config)
            scheduler = UpdateScheduler(optimizer, math.ceil(65931 / 128))
            restore_checkpoint(tx, commit, restored, optimizer, scheduler, config, identity['scientific_config_sha256'])
            assert tensor_hash(restored.state_dict().items()) == tensor_hash(payload['model'].items())
            assert same(payload['optimizer'], optimizer.state_dict())
            assert scheduler.updates == len(rows) * 516
            effective_update(restored, optimizer, scheduler, [batches[0]], device)
            hashes.append(tensor_hash(restored.state_dict().items()))
            optimizers.append(cpu_tree(optimizer.state_dict()))
            del restored, optimizer
        assert hashes[0] == hashes[1] and same(optimizers[0], optimizers[1])
        results[run_id] = {'restored_completed_epochs': len(rows),
            'checkpoint_source_commit': payload['execution_commit'],
            'scientific_config_sha256': identity['scientific_config_sha256'],
            'restored_full_state_and_optimizer_equal': True, 'repeat_next_update_bitwise_equal': True,
            'production_checkpoint_modified': False}
        del payload, optimizers
        torch.cuda.empty_cache()
    for key in ['r50', 'vitb16']:
        config = next(c for c in candidates() if c['dataset'] == 'svhn' and c['model_key'] == key and
                      c['method'] == 'adapter' and c['tuning_candidate_index'] == 2)
        identity = scientific_identity(config)
        model, _ = build_model(config, device)
        optimizer, _ = make_optimizer(model, config)
        scheduler = UpdateScheduler(optimizer, math.ceil(65931 / 128))
        effective_update(model, optimizer, scheduler, [batches[0]], device)
        model.eval()
        with torch.no_grad():
            reference_logits = model(batches[0][0].to(device)).cpu()
        from tlcls.models import apply_strategy
        apply_strategy(model, key, 'adapter', True)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                   'rng': rng_state(), 'config': config, 'completed_epoch': 1,
                   'scientific_config_sha256': identity['scientific_config_sha256']}
        with tempfile.TemporaryDirectory(prefix='P2_gpu_transaction_', dir=str(STAGE)) as folder:
            tx = EpochTransaction(folder, identity['scientific_config_sha256'])
            value, _ = tx.commit(1, {'epoch': 1}, lambda f: torch.save(payload, f),
                                lambda f: torch.save({'model': model.state_dict()}, f), 1, 1)
            for batch in batches[1:]:
                effective_update(model, optimizer, scheduler, [batch], device)
            expected_model = tensor_hash(model.state_dict().items())
            expected_optimizer = cpu_tree(optimizer.state_dict())
            restored, _ = build_model(config, device)
            restored_optimizer, _ = make_optimizer(restored, config)
            restored_scheduler = UpdateScheduler(restored_optimizer, math.ceil(65931 / 128))
            restore_checkpoint(tx, value, restored, restored_optimizer, restored_scheduler,
                               config, identity['scientific_config_sha256'])
            restored.eval()
            with torch.no_grad():
                actual_logits = restored(batches[0][0].to(device)).cpu()
            self_error = float((actual_logits - reference_logits).abs().max())
            assert self_error == 0
            apply_strategy(restored, key, 'adapter', True)
            for batch in batches[1:]:
                effective_update(restored, restored_optimizer, restored_scheduler, [batch], device)
            assert tensor_hash(restored.state_dict().items()) == expected_model
            assert same(expected_optimizer, restored_optimizer.state_dict())
            # 元数据 LR 被篡改时必须先拒绝恢复，不能加载旧 optimizer 后继续训练。
            altered = dict(config, learning_rate=config['learning_rate'] * 10)
            rejected = False
            try:
                restore_checkpoint(tx, value, restored, restored_optimizer, restored_scheduler,
                                   altered, scientific_identity(altered)['scientific_config_sha256'])
            except AssertionError:
                rejected = True
            assert rejected
            results[key] = {'eval_logit_max_abs_error': self_error,
                'continued_full_model_bitwise_equal': True, 'continued_optimizer_state_bitwise_equal': True,
                'scientific_LR_mismatch_rejected': True, 'checkpoint': value['last'],
                'temporary_acceptance_checkpoint_removed_after_pass': True}
        del model, optimizer, restored, restored_optimizer, expected_optimizer, payload
        torch.cuda.empty_cache()
    write(STAGE / 'gpu_acceptance.json', {'status': 'passed', 'results': results,
        'execution_commit': code_commit(), 'execution_identity': execution_identity(),
        'authorization_sha256': file_hash(STAGE / 'authorization.json'),
        'parent_attempt_id': os.environ['TLCLS_ATTEMPT_ID'], 'test_access': False})


if __name__ == '__main__':
    main()
