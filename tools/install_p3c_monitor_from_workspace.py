"""配置30分钟检查；优先工具不可用时沿用本机已使用的heartbeat文件格式。"""
import json
import os
from pathlib import Path
import textwrap
import time
import tomllib
from remote_session import connect
from transport import ROOT,upload,command
from poll_p3c import SCRIPT

def main():
    source='import sys,time,json,fcntl,os\nfrom pathlib import Path\nsys.path.insert(0,"/root/user_data/tl_cls_v1/src")\nfrom tlcls.p3_core import write,process_ticks\n\ndef check():\n'
    source+=textwrap.indent(SCRIPT.replace('print(json.dumps(v))','return v'),'    ')
    source+='''
def main():
    folder=Path('/root/user_data/tl_cls_v1/records/p3c/monitor');folder.mkdir(exist_ok=True)
    handle=(folder/'monitor.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(folder/'registration.json',{'pid':os.getpid(),'start_ticks':process_ticks(os.getpid()),'interval_seconds':1800})
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:value=check()
        except Exception as error:value={'actionable_reason':'monitor_read_failed','error_type':type(error).__name__}
        write(folder/'latest.json',value)
        with (folder/'checks.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\\n')
        if value.get('actionable_reason'):
            write(folder/'attention.json',value);return
        time.sleep(1800)
if __name__=='__main__':main()
'''
    target=ROOT/'p3_source/tools/p3c_monitor_server.py';target.write_text(source,encoding='utf8',newline='\n')
    client=connect()
    try:
        upload(client,target,'tools/p3c_monitor_server.py')
        script=r'''import pathlib,sys,subprocess,time,json
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import read,write,process_alive,process_ticks
s=r/'records/p3c/monitor';s.mkdir(exist_ok=True)
registered=read(s/'registration.json') if (s/'registration.json').exists() else None
if registered and process_alive(registered['pid'],registered['start_ticks']):
 print(json.dumps({'existing_monitor_reused':True,**registered}))
else:
 with (s/'monitor.log').open('ab') as log:
  p=subprocess.Popen([sys.executable,str(r/'tools/p3c_monitor_server.py')],cwd=r,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
 v={'pid':p.pid,'start_ticks':process_ticks(p.pid),'interval_seconds':1800,'started_unix':time.time(),'training_or_evaluation_restarted_by_monitor':False}
 write(s/'launch.json',v);print(json.dumps(v))
'''
        code,out,_=command(client,['/root/miniconda3/envs/tl_cls_v1/bin/python','-c',script]);assert code==0
        server=json.loads(out)
    finally:client.close()
    identity='tlcls-p3c-30min';directory=Path.home()/'.codex/automations'/identity;directory.mkdir(parents=True,exist_ok=True)
    prompt='''继续本线程已授权P3-C集中测试监督，每次只检查一次，不运行前台watch。工作目录E:/图像分类迁移学习；先运行python ops/poll_p3c.py。用户PI已明确取消独立签署要求，授权代理安全配置密钥并执行冻结400项官方test、后台运行、完成后回收分析推送；不要再次索要同一授权。密钥仅在服务器/run运行时0600文件及进程环境中，禁止回传、回显、写入日志或Git；ops/remote_session.py负责受控连接，不写真实连接地址和口令。固定项目/root/user_data/tl_cls_v1，环境/root/miniconda3/envs/tl_cls_v1，真实封存源码HEAD700a1adc不变，4份tools/p3c运行源码由签名授权及CPU回执绑定。P3-C新增GPU总额硬上限3小时，包含合成GPU验收/加载/门禁/推理/写盘/失败/重试/退出；同一shared P3账本总上限160小时，旧P3B92次attempt保留；旧seal账本SHA对应records/p3c/p3b_gpu_ledger.sealed.json，不重写旧seal。运行顺序80best→80last→240prefix，不改模型、LR、B、checkpoint或五种子。不在运行中按测试成绩调整计划。正常且无可操作变化保持安静，只在完成、失败、暂停、预算/资源异常或需要用户处理时通知。普通授权内运维故障先保全原日志和源码，再安全离线排查、最小修复及必要回归；不因工具/连接故障要求重复批准。目标状态不明时停止危险目标动作，继续安全离线工作。禁止盲重启、扩大3/160预算或按分数重跑；科学身份/模型损坏/预算不足须停发新任务并如实报告。完成400项且所有guardian结算后，确认records/p3c/independent_delivery_audit.json通过和P3C_sealed.json；运行python ops/collect_p3c.py回收小型文本/指标，所有NPZ原始预测和checkpoint留服务器，不回收密钥。独立重算来源/配对统计/账本，生成正式测试中文汇报、主次/时间切片表、预注册五seed paired t95%区间与双侧检验Holm、CIFAR/ViT LP−FT -0.5pp非劣规则、探索性效率图和审计。复用Python/nature-figure流程，原始预测不上传公开仓库；推送Serendipity-min/image-classification-transfer-learning的codex/p3-execution受保护分支。只有完整证据才可PASS，未完成报告已完成/缺失ID和实际费用。完成交付或真正需人工裁定的阻塞后将本automation.toml status改为PAUSED，不归档聊天。训练、扩展实验和论文定稿仍未授权。'''
    prompt+='完整test_authorization.json及运行凭据引用保留服务器或受控本地，不推送公开仓库；公开交付授权SHA、模式和校验回执即可。'
    path=directory/'automation.toml';now=int(time.time()*1000)
    data={'version':1,'id':identity,'kind':'heartbeat','name':'P3-C集中测试每30分钟检查','prompt':prompt,
        'status':'ACTIVE','rrule':'FREQ=MINUTELY;INTERVAL=30','target_thread_id':os.environ['CODEX_THREAD_ID'],
        'created_at':now,'updated_at':now}
    if path.exists():data['created_at']=tomllib.loads(path.read_text(encoding='utf8'))['created_at']
    content='\n'.join(k+' = '+(json.dumps(v,ensure_ascii=False) if isinstance(v,str) else str(v)) for k,v in data.items())+'\n'
    path.write_text(content,encoding='utf8',newline='\n');assert tomllib.loads(path.read_text(encoding='utf8'))==data
    receipt={'desktop_automation':identity,'status':'ACTIVE','readback_verified':True,'interval_minutes':30,
        'server_monitor':server,'normal_notifications':'quiet','no_foreground_watch':True}
    (ROOT/'records/p3c/monitor_registration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('P3C_monitor_install_failed:'+type(error).__name__)
        raise SystemExit(1)
