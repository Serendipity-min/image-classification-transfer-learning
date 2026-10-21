"""P0 只读环境盘点；只创建本项目目录，不安装依赖、不联网、不启动训练。"""

import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path("/root/user_data/tl_cls_v1")
for folder in ("records", "reports", "configs", "src", "data", "weights", "runs"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


def run(command, timeout=45):
    # 不执行 shell，输出仅限本脚本明确列出的设备和版本信息。
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"error_type": type(exc).__name__}


def read(path):
    file = Path(path)
    return file.read_text().strip() if file.is_file() else None


conda = Path("/root/miniconda3")
interpreters = [conda / "bin/python"]
if (conda / "envs").is_dir():
    interpreters += sorted((conda / "envs").glob("*/bin/python"))
probe = '''import json,sys,importlib.metadata as m
names=['torch','torchvision','timm','numpy','scipy','Pillow','huggingface-hub','safetensors','psutil','pip']
versions={}
for name in names:
    try: versions[name]=m.version(name)
    except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':sys.version.split()[0],'packages':versions}))
'''
environments = []
for python in interpreters:
    if python.is_file():
        environments.append({"python_path": str(python), "probe": run([str(python), "-c", probe])})

usage = shutil.disk_usage(ROOT)
receipt = {
    "protocol_id": "TL-CLS-20261020-v1.0",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "project_root": str(ROOT),
    "stage": "P0_environment_inventory",
    "environment_installation_authorized": False,
    "network_access": "offline_per_user_no_network_requests_made",
    "gpu": run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version,power.limit,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]),
    "gpu_processes": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]),
    "cpu": run(["lscpu"]),
    "quota": {p: read(p) for p in ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]},
    "disk": {"filesystem_total_bytes": usage.total, "filesystem_free_bytes": usage.free,
             "tenant_confirmed_quota_bytes": None, "planning_cap_gib": 100},
    "environments": environments,
    "existing_project_assets": {folder: [str(x.relative_to(ROOT)) for x in (ROOT / folder).glob("*")]
                                for folder in ["data", "weights", "src", "runs"]},
    "p0_acceptance": "pending", "p1_completed_runs": 0,
    "training_results": None,
}
# 原子替换，避免连接断开留下不完整收据；未知科学字段保持 null/pending。
destination = ROOT / "records/server_preflight.json"
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
temporary.replace(destination)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
