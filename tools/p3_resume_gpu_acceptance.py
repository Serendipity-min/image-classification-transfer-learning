"""真实 P3 中断 checkpoint 的只读 GPU 恢复验收；不写回正式 run。"""
import math
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from tlcls.common import setup, tensor_hash
from tlcls.engine import UpdateScheduler, effective_update
from tlcls.models import build_model, make_optimizer
from tlcls.p3_budget import Budget
from tlcls.p3_core import (ROOT, STAGE, candidates, code_commit, execution_identity,
    scientific_identity, file_hash, read, write)
from tlcls.p3_transaction import EpochTransaction
from p3_train import restore_checkpoint
from p3_gpu_acceptance import same, cpu_tree


def main():
    attempt = Budget().snapshot()['attempts'][os.environ['TLCLS_ATTEMPT_ID']]
    assert attempt['kind'] == 'gpu_acceptance'
    device_index = attempt['device']
    run_id = ['P3-cifar100-r50-scratch-s2026', 'P3-cifar100-r50-full_ft-s42'][device_index]
    config = next(r for r in candidates() if r['run_id'] == run_id)
    identity = scientific_identity(config)
    directory = ROOT / 'runs' / run_id
    before = {n: file_hash(directory / n) for n in ['current.json', 'epochs.jsonl', 'summary.json', 'prefixes.json']}
    tx = EpochTransaction(directory, identity['scientific_config_sha256'], config)
    current, rows = tx.load(rebuild=False)
    assert len(rows) == [3, 37][device_index]
    torch.cuda.set_device(device_index); setup(config['seed'])
    device = torch.device('cuda', device_index)
    generator = torch.Generator().manual_seed(20261026)
    batch = (torch.randn(2, 3, 224, 224, generator=generator), torch.tensor([1, 2]),
             ['resume_acceptance/synthetic/0', 'resume_acceptance/synthetic/1'])
    hashes, state = [], None
    for _ in range(2):
        model, _ = build_model(config, device)
        optimizer, _ = make_optimizer(model, config)
        scheduler = UpdateScheduler(optimizer, math.ceil(45000 / 128))
        payload = restore_checkpoint(tx, current, model, optimizer, scheduler, config, identity['scientific_config_sha256'])
        assert tensor_hash(model.state_dict().items()) == tensor_hash(payload['model'].items())
        assert same(optimizer.state_dict(), payload['optimizer'])
        assert scheduler.updates == len(rows) * 352
        effective_update(model, optimizer, scheduler, [batch], device)
        hashes.append(tensor_hash(model.state_dict().items()))
        now = cpu_tree(optimizer.state_dict())
        if state is not None: assert same(state, now)
        state = now
        del payload, model, optimizer
        torch.cuda.empty_cache()
    assert hashes[0] == hashes[1]
    assert before == {n: file_hash(directory / n) for n in before}
    write(STAGE / f'resume_gpu_acceptance_{device_index}.json', {'status': 'passed',
        'run_id': run_id, 'restored_completed_epochs': len(rows), 'production_files_modified': False,
        'full_model_optimizer_scheduler_RNG_restored': True, 'repeat_next_update_bitwise_equal': True,
        'checkpoint': current['last'], 'execution_commit': code_commit(),
        'execution_identity': execution_identity(), 'parent_attempt_id': attempt['attempt_id'],
        'acceptance_tool_sha256': file_hash(Path(__file__)), 'official_test_access': False})


if __name__ == '__main__': main()
