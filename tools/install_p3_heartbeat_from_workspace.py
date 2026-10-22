"""沿用已存在的本机 heartbeat 存储格式；每次只检查一次，正常运行保持安静。"""
import json
import os
from pathlib import Path
import time
import tomllib


def main():
    identity = 'tlcls-p3-30min'
    directory = Path.home() / '.codex/automations' / identity
    path = directory / 'automation.toml'
    thread = os.environ['CODEX_THREAD_ID']
    prompt = '''继续本线程已授权的 TL-CLS P3 合同监督，每次只检查一次，不运行前台 watch。工作目录 E:/图像分类迁移学习；先运行 python ops/poll_p3.py，检查 records/p3/monitor/latest.json、服务器 records/p3/gpu_ledger.json、阶段状态、cost_calibration 和 storage_latest。只通过 ops/remote_session.py 的受控运行时连接；真实服务器地址、口令、私钥不得写入脚本、日志或 Git。服务器工作路径固定 /root/user_data/tl_cls_v1，环境 /root/miniconda3/envs/tl_cls_v1。P2 987fb5a 归档只读，不重跑 P2；P3 执行实际提交以 records/p3/acceptance.json 和 run_manifest 为准，禁止未经审查的跨代码恢复。80 正式运行，5 seeds=17,42,2026,3407,65537，每 run 全新初始化 50 epochs、5 warmup，已选 LR 和四组 timing-only B 不可变。P3 总硬上限 160 GPU-hours，包含验收、失败、恢复、诊断和退出；测试预留不可转用；200 GiB 人工分配保障，175 GiB 预警，安全余量10 GiB加实际原子保存空间。正常运行且无可操作变化时保持安静，不发例行状态。只在完成、失败、暂停、预算/存储/身份异常或需要用户处理时通知。先保留和分析异常证据，不自动扩预算、改科学参数或重启。只有80个run全部终态且所有guardian结算、独立审计通过，才能生成或确认 records/P3_sealed.json 和冻结 evaluation_manifest 草案；不完整必须明确标记，不可报通过。按本线程授权回收小型文本证据并核验 SHA，checkpoint 留服务器，不回收大型文件；生成真实验证结果曲线、GPU/存储/attempt/checkpoint台账和中文阶段汇报，推送既有私有GitHub仓库 codex/p3-execution 分支，回交PI审核。P3-C、官方test标签、测试预测、测试图、扩展实验和发表始终保持关闭，封存不自动授权测试。完成交付或遇到需人工裁定的阻塞后将本 automation.toml status 改为 PAUSED；不要归档聊天。'''
    now = int(time.time() * 1000)
    created = now
    if path.exists():
        old = tomllib.loads(path.read_text(encoding='utf8'))
        assert old['target_thread_id'] == thread
        created = old['created_at']
    data = {'version': 1, 'id': identity, 'kind': 'heartbeat', 'name': 'P3 实验每30分钟检查',
        'prompt': prompt, 'status': 'ACTIVE', 'rrule': 'FREQ=MINUTELY;INTERVAL=30',
        'target_thread_id': thread, 'created_at': created, 'updated_at': now}
    text = '\n'.join(k + ' = ' + (json.dumps(v, ensure_ascii=False) if isinstance(v, str) else str(v))
                     for k, v in data.items()) + '\n'
    assert tomllib.loads(text) == data
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'automation.toml.tmp'
    temporary.write_text(text, encoding='utf8'); temporary.replace(path)
    print(json.dumps({'id': identity, 'status': 'ACTIVE', 'interval_minutes': 30,
                      'readback_verified': tomllib.loads(path.read_text(encoding='utf8')) == data}))


if __name__ == '__main__': main()
