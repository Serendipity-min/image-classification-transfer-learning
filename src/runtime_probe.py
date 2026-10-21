"""验证现有 CUDA 栈并清点离线依赖；不下载、不安装。"""

import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

import torch
import torchvision


ROOT = Path("/root/user_data/tl_cls_v1")
names = ["torch", "torchvision", "numpy", "scipy", "Pillow", "psutil", "timm",
         "huggingface-hub", "safetensors", "PyYAML", "filelock", "fsspec",
         "requests", "packaging", "tqdm", "typing_extensions", "setuptools",
         "urllib3", "certifi", "charset-normalizer", "idna"]
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

# 仅执行小张量功能检查，不能替代后续模型确定性、数值与恢复验收。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
devices = []
for device in range(torch.cuda.device_count()):
    with torch.cuda.device(device):
        tensor = torch.ones((32, 32), device=f"cuda:{device}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = tensor @ tensor
        torch.cuda.synchronize(device)
        devices.append({"index": device, "bf16_supported": torch.cuda.is_bf16_supported(),
                        "matmul_dtype": str(value.dtype),
                        "matmul_finite": bool(torch.isfinite(value).all().item()),
                        "matmul_expected_value": bool((value == 32).all().item())})

receipt = {"protocol_id": "TL-CLS-20261020-v1.0",
           "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "packages": packages, "torch_cuda": torch.version.cuda,
           "cudnn_version": torch.backends.cudnn.version(), "devices": devices,
           "torchvision_ops_check": torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]),
                                                        torch.tensor([1.]), 0.5).tolist(),
           "existing_asset_candidates": {},
           "environment_installation_authorized": False,
           "model_acceptance": "pending_dependencies_and_assets"}
# 仅查看约定缓存与用户数据目录，避免读取其他项目文件内容。
for location in ["/root/.cache/huggingface/hub", "/root/.cache/torch/hub/checkpoints", "/root/user_data"]:
    directory = Path(location)
    receipt["existing_asset_candidates"][location] = (
        [p.name for p in directory.iterdir()][:80] if directory.is_dir() else [])
destination = ROOT / "records/runtime_probe.json"
destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
# pip freeze 可能包含私人源地址，只保留 name==version 的包清单。
lock = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()
              if dist.metadata.get("Name"))
(ROOT / "records/installed_versions.txt").write_text("\n".join(lock) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
