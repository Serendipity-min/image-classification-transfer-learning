"""协议、随机流、原子文件与证据身份的公共实现。"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

# 必须在首次 CUDA 上下文创建前设置，保证严格确定性矩阵乘法可用。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch

ROOT = Path(os.environ.get('TLCLS_ROOT', '/root/user_data/tl_cls_v1'))


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def derive_seed(*fields):
    # 冻结编码为无空格 JSON 数组，SHA-256 整数低 64 位，跨进程不使用 hash()。
    raw = json.dumps(fields, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[-8:], 'big')


def setup(seed, deterministic=True):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    # 单任务仅使用已绑定的这一张卡，不为收集未使用 RNG 而初始化第二张 GPU。
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(torch.cuda.current_device()),
            'cuda_rng_scope': 'current_bound_device_only'}


def restore_rng(state):
    # 构造模型/优化器并加载状态后，紧邻下一批数据前调用。
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], torch.cuda.current_device())


@contextlib.contextmanager
def rng_stream(seed):
    previous = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng(previous)


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def code_commit():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(dataset=None, model_key=None):
    assets_path = ROOT / 'records/assets.json'
    assets = load_json(assets_path)['assets'] if assets_path.exists() else []
    split = ROOT / 'records' / f'{dataset}_split.json'
    return {'protocol_id': 'TL-CLS-20261020-v1.0',
            'protocol_sha256': sha256(ROOT / 'configs/protocol.json'),
            'code_commit': code_commit(),
            'split_sha256': sha256(split) if split.exists() else None,
            'weight_sha256': next((x['sha256'] for x in assets if x['key'] == f'{model_key}_model.safetensors'), None),
            'dependency_lock_sha256': sha256(ROOT / 'records/dependencies.lock') if (ROOT / 'records/dependencies.lock').exists() else None}


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def finite_or_raise(tensor, location):
    mask = torch.isfinite(tensor)
    if not bool(mask.all().item()):
        values = tensor.detach()[mask]
        error = FloatingPointError(location)
        # 异常时只统计有限子集，不对坏 logits 执行 argmax 伪造准确率。
        error.diagnostics = {'location': location, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                             'non_finite_elements': int((~mask).sum()),
                             'finite_min': float(values.min()) if values.numel() else None,
                             'finite_max': float(values.max()) if values.numel() else None}
        raise error


def sync_time():
    torch.cuda.synchronize()
    return time.monotonic()
