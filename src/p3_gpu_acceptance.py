"""两设备、两策略的合成输入 GPU 事务验收；不读官方 test，实际时间记 P3。"""
import math
import os
import tempfile
import torch
from tlcls.common import rng_state, setup, tensor_hash
from tlcls.engine import UpdateScheduler, effective_update
from tlcls.models import apply_strategy, build_model, make_optimizer
from tlcls.p3_budget import Budget
from tlcls.p3_core import (STAGE, candidates, code_commit, execution_identity, file_hash,
                           read, require_authorization, scientific_identity, write)
from tlcls.p3_transaction import EpochTransaction
from p3_train import restore_checkpoint


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def same(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    require_authorization(False)
    cpu = read(STAGE / 'cpu_acceptance.json')
    assert cpu['status'] == 'passed' and cpu['execution_identity'] == execution_identity()
    attempt = Budget().snapshot()['attempts'][os.environ['TLCLS_ATTEMPT_ID']]
    assert attempt['kind'] == 'gpu_acceptance'
    index = attempt['device']
    torch.cuda.set_device(index); setup(17)
    device = torch.device('cuda', index)
    key, method = ('r50', 'full_ft') if index == 0 else ('vitb16', 'adapter')
    config = next(c for c in candidates() if c['dataset'] == 'cifar100' and
                  c['model_key'] == key and c['method'] == method and c['seed'] == 17)
    identity = scientific_identity(config)
    generator = torch.Generator().manual_seed(17042)
    batches = [(torch.randn(2, 3, 224, 224, generator=generator), torch.tensor([1, 2]),
                ['P3_acceptance/synthetic/a', 'P3_acceptance/synthetic/b']) for _ in range(3)]
    model, model_identity = build_model(config, device)
    optimizer, groups = make_optimizer(model, config)
    scheduler = UpdateScheduler(optimizer, math.ceil(45000 / 128))
    effective_update(model, optimizer, scheduler, [batches[0]], device)
    model.eval()
    with torch.no_grad(): reference = model(batches[0][0].to(device)).cpu()
    apply_strategy(model, key, method, True)
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'rng': rng_state(), 'config': config, 'completed_epoch': 1,
        'execution_commit': code_commit(), 'runtime_execution_identity': execution_identity(),
        'scientific_config_sha256': identity['scientific_config_sha256']}
    with tempfile.TemporaryDirectory(prefix='P3_gpu_transaction_', dir=STAGE) as folder:
        tx = EpochTransaction(folder, identity['scientific_config_sha256'], config)
        value, _ = tx.commit(1, {'epoch': 1}, lambda f: torch.save(payload, f),
            lambda f: torch.save({'model': model.state_dict(), 'config': config}, f), 1,
            lambda: Budget().run_seconds(attempt['run_id']))
        for batch in batches[1:]: effective_update(model, optimizer, scheduler, [batch], device)
        expected_model = tensor_hash(model.state_dict().items())
        expected_optimizer = cpu_tree(optimizer.state_dict())
        del model, optimizer, payload
        torch.cuda.empty_cache()
        restored, _ = build_model(config, device)
        restored_optimizer, _ = make_optimizer(restored, config)
        restored_scheduler = UpdateScheduler(restored_optimizer, math.ceil(45000 / 128))
        restore_checkpoint(tx, value, restored, restored_optimizer, restored_scheduler,
                           config, identity['scientific_config_sha256'])
        restored.eval()
        with torch.no_grad(): actual = restored(batches[0][0].to(device)).cpu()
        error = float((actual - reference).abs().max()); assert error == 0
        apply_strategy(restored, key, method, True)
        for batch in batches[1:]: effective_update(restored, restored_optimizer, restored_scheduler, [batch], device)
        assert tensor_hash(restored.state_dict().items()) == expected_model
        assert same(expected_optimizer, restored_optimizer.state_dict())
        altered = dict(config, learning_rate=config['learning_rate'] * 10)
        try:
            restore_checkpoint(tx, value, restored, restored_optimizer, restored_scheduler,
                               altered, scientific_identity(altered)['scientific_config_sha256'])
        except AssertionError: pass
        else: raise AssertionError('LR mismatch unexpectedly accepted')
        # 额外验证错误执行代码元数据，在加载 optimizer 之前必须拒绝。
        corrupt = torch.load(tx.path(value['last']['path']), weights_only=False, map_location='cpu')
        corrupt['execution_commit'] = 'synthetic-invalid-commit'
        torch.save(corrupt, tx.path(value['last']['path']))
        try:
            restore_checkpoint(tx, value, restored, restored_optimizer, restored_scheduler,
                               config, identity['scientific_config_sha256'])
        except AssertionError: pass
        else: raise AssertionError('code mismatch unexpectedly accepted')
    write(STAGE / f'gpu_acceptance_{index}.json', {'status': 'passed', 'device': index,
        'model_key': key, 'method': method, 'seed': 17, 'eval_logit_max_abs_error': error,
        'continued_model_bitwise_equal': True, 'continued_optimizer_bitwise_equal': True,
        'LR_and_code_mismatch_rejected': True, 'model_identity': model_identity,
        'execution_commit': code_commit(), 'execution_identity': execution_identity(),
        'parent_attempt_id': attempt['attempt_id'], 'official_test_access': False,
        'all_inputs_synthetic': True, 'temporary_checkpoints_removed': True})


if __name__ == '__main__':
    main()
