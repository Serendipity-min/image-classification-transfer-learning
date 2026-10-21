"""固定实际数据工作量的独占/并发微基准，不生成额外正式实验。"""

import argparse
import itertools
import json
import math
import os
import time

import torch

from tlcls.common import ROOT, atomic_json, load_json, provenance, setup, sync_time
from tlcls.data import TrainValData, loader
from tlcls.engine import UpdateScheduler, effective_update
from tlcls.models import build_model, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--barrier-unix', type=float, default=0)
    args = parser.parse_args()
    started = float(os.environ.get('TLCLS_PROCESS_STARTED_MONOTONIC', time.monotonic()))
    torch.cuda.set_device(args.device)
    setup(314159)
    device = torch.device('cuda', args.device)
    config = next(r for r in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['run_id'] == args.run_id)
    model, _ = build_model(config, device)
    optimizer, _ = make_optimizer(model, config)
    dataset = TrainValData(config['dataset'], config['model_key'], 'train', 314159)
    dataloader, _ = loader(dataset, 314159, True)
    stream = iter(dataloader)
    scheduler = UpdateScheduler(optimizer, math.ceil(len(dataset) / 128))
    # 两次 warmup 后再测 6 个有效 batch；数据次序、增强和更新数保持固定。
    for _ in range(2):
        effective_update(model, optimizer, scheduler, [next(stream) for _ in range(4)], device)
    while time.time() < args.barrier_unix:
        time.sleep(min(0.2, args.barrier_unix - time.time()))
    tick = sync_time()
    for _ in range(6):
        effective_update(model, optimizer, scheduler, [next(stream) for _ in range(4)], device)
    seconds = sync_time() - tick
    output = {'run_id': args.run_id, 'label': args.label, 'device': args.device,
              'provenance': provenance(config['dataset'], config['model_key']),
              'measured_effective_updates': 6, 'measured_samples': 768,
              'seconds': seconds, 'images_per_second': 768 / seconds,
              'gpu_reserved_seconds': time.monotonic() - started,
              'workers': 4, 'uses_real_training_data': True,
              'status': 'completed', 'formal_run': False}
    atomic_json(ROOT / 'records' / f'benchmark_{args.label}_{args.device}.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
