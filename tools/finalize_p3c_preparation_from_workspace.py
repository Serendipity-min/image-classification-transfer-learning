"""整理真实准备回执与未签署请求；绝不生成 PI 密钥或真实签名。"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT/'p3_source'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')

def main():
    stage=REPO/'records/p3c';stage.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'records/p3c').glob('*'):
        if p.is_file() and p.suffix in ['.json','.log']:shutil.copyfile(p,stage/p.name)
    cpu=read(stage/'fault_injection.json');entry=read(stage/'entry_preflight.json')
    assert cpu['status']=='passed' and cpu['tests_run']>=27 and entry['actual_unsigned_launch_denied']
    assert entry['production_gpu_ledger_unchanged_after_acceptance']
    sources={f'tools/{name}':sha(REPO/'tools'/name) for name in cpu['source_sha256']}
    assert {Path(n).name:h for n,h in sources.items()}==cpu['source_sha256']
    seal=read(REPO/'records/P3_sealed.json')
    contract=ROOT/'doc/TL-CLS_P3B审查及P3C集中测试阶段合同_20261027.md'
    shutil.copyfile(contract,REPO/'docs/P3_C_集中测试阶段合同_20261027.md')
    payload={'schema':'P3_C_authorization_v1','separate_user_approval':True,
        'PI_independent_issuance':True,'P3_C_authorized':True,'centralized_test_authorized':True,
        'approval_id':'P3C-20261027-'+uuid.uuid4().hex,
        'seal_sha256':sha(REPO/'records/P3_sealed.json'),
        'evaluation_manifest_sha256':seal['evaluation_manifest_sha256'],
        'selected_hparams_sha256':seal['selected_hparams_sha256'],
        'execution_identity_sha256':seal['execution_identity_sha256'],
        'execution_commit':seal['execution_commit'],
        'P3_B_gpu_ledger_sha256':seal['gpu_ledger_sha256'],
        'CPU_fault_injection_sha256':sha(stage/'fault_injection.json'),
        'execution_order_sha256':sha(stage/'execution_order.json'),
        'supervisor_source_sha256':sources,'subcap_gpu_seconds':10800,'total_cap_gpu_seconds':576000,
        'scope':'synthetic GPU acceptance then fixed400 evaluations, independent analysis; no training or retuning',
        'contract_sha256':sha(contract),'created_request_unix':time.time()}
    write(stage/'authorization_request.json',{'status':'unsigned_request_NOT_AUTHORIZATION',
        'warning':'Only PI-controlled independent signing makes the payload effective; no real signature or key is present',
        'payload':payload})
    write(stage/'independent_authorization_receipt.json',{'status':'NOT_ISSUED',
        'test_authorization_present_on_server':False,'PI_runtime_key_injected':False,
        'request_sha256':sha(stage/'authorization_request.json'),'official_test_accessed':False})
    write(stage/'gpu_acceptance.json',{'status':'NOT_RUN','reason':'awaiting independently signed scope and external runtime key','GPU_seconds':0})
    gates={'G1':'ACCEPTED_PRIOR_HASH_EVIDENCE_CURRENT_508_FILES_AND_SIZES',
        'G2':'BLOCKED_INDEPENDENT_SIGNATURE_AND_KEY_NOT_AVAILABLE',
        'G3':f'CPU_{cpu["tests_run"]}_PASS_GPU_NOT_RUN',
        'G4':'DUAL_BUDGET_AND_SIGNED_APPEND_IMPLEMENTED_CPU_TESTED_PRODUCTION_NOT_STARTED',
        'G5':'PASS_FROZEN_80_BEST_80_LAST_240_PREFIX_ORDER',
        'G6':'PASS_SYNTHETIC_METRICS_AND_PAIRED_STATISTICS_REAL_TEST_NOT_RUN'}
    write(stage/'preparation_audit.json',{'status':'PREPARATION_COMPLETED_TEST_NO_GO',
        'gates':gates,'checkpoint_rehash_waived_by_current_user':True,'new_GPU_seconds':0,
        'P3_B_gpu_hours_unchanged':124.61060950655697,'P3_C_authorized':False,
        'centralized_test_authorized':False,'CPU_fault_injection_sha256':sha(stage/'fault_injection.json'),
        'supervisor_source_sha256':sources,'PI_unsigned_request_sha256':sha(stage/'authorization_request.json'),
        'official_test_predictions':0,'source_inside_seal_modified':False})
    document=f'''# P3-C 启动准备与待签署回执

2026-10-27。**工程准备已完成；正式集中测试 NO_GO，尚未启动。** P3-B 按用户新合同接受，不重训，不更换LR、checkpoint或B。当前用户明确表示服务器未变化、不重复复核哈希；因此复用上轮已接受的真实二进制哈希回执，当前只确认508个文件存在且大小一致，未再次逐字节读取全部模型。正式执行中的原始严格准入函数仍保留。

|门|当前状态|
|---|---|
'''+''.join(f'|{k}|{v}|\n' for k,v in gates.items())+f'''
实际服务器HEAD继续为700a1adc；本次仅新增tools运维程序，没有checkout交付commit，没有修改seal覆盖的18个src文件、P3-B seal或80个summary。新程序SHA与真实CPU验收绑定，未来独立签署也绑定这些SHA。

CPU合成验收 **{cpu['tests_run']}/{cpu['tests_run']} PASS**，覆盖原子NPZ及收据提交、中断三边界、重复启动、损坏输出拒绝、错checkpoint/manifest/路径/签名拒绝、空间不足、两卡互斥、历史账本保全、真实CPU子进程在父进程死亡/预算条件下退出、指标重算及固定五seed配对t/95%区间/Holm/非劣界。实际未签署入口拒绝执行；所有测试均为临时目录与CPU合成输入，官方TestData未构造，新增GPU成本为0。

执行顺序已独立冻结：80 primary_best_val → 80 secondary_last → 240 budget_prefix，各类内按run_id及fraction稳定排序，前一类完成后才发后一类。恢复仅复用checkpoint/manifest/预测SHA均通过的completed收据，半成品不算完成，技术恢复保留旧调度证据。

监督器按每卡一个独立guardian控制工作进程组，父进程退出、失去进度、磁盘不足、预测或实际超支都会停止发新任务并退出结算。3 GPU-hours子上限和原160 GPU-hours总封顶同时生效，包含GPU验收、严格准入、加载、推理、压缩写入、失败与退出。3小时是两卡占用之和。原P3-B账本字节另留sealed快照，新费用追加到同一shared ledger，并用独立HMAC追加链记录；不改写旧92个attempt或旧seal引用。

原始NPZ预测和逐项混淆矩阵将保存在服务器records/p3c/results；不会自动上传公开仓库。验收后分析程序从预测独立重算Top-1、正确数、macro-F1、balanced accuracy和NLL，再对两项固定主比较做五seed配对区间、两项双侧t的Holm校正及预设−0.5pp非劣规则。缺seed不作完整主推断。所有正式测试、效率图和实际结果目前均NOT_RUN，不提前提供测试成绩。

## 当前真实阻塞

合同G2明确要求“由与训练执行账户分离的授权流程”签发授权，并规定“密钥由PI控制”。服务器当前既无records/p3/test_authorization.json，也没有外部注入的TLCLS_TEST_APPROVAL_HMAC_KEY。聊天中的执行指令没有提供独立签发材料；本次没有替PI生成密钥、签名或把separate_user_approval伪造成已生效授权。

待签署包：[authorization_request.json](../records/p3c/authorization_request.json)，请求SHA `{sha(stage/'authorization_request.json')}`。它明确标为unsigned，不能直接当测试授权。请由PI控制的独立流程签发，并通过外部运行环境注入同一HMAC密钥；**不要在聊天、仓库或命令行粘贴真实密钥**。具体辅助工具和启动方式见[独立签署与启动说明](P3_C_PI独立签署与启动.md)。独立签署完成后，先做双卡合成GPU入口验收；通过后才允许400项官方集中测试，若预算不足则交回完整未完成清单，不追加额度。

## 交付

- [真实入口回执](../records/p3c/entry_preflight.json)、[CPU故障验收](../records/p3c/fault_injection.json)、[逐测试日志](../records/p3c/fault_injection.log)、[准备审计](../records/p3c/preparation_audit.json)。
- [冻结执行顺序](../records/p3c/execution_order.json)、[零新增费用子账本](../records/p3c/evaluation_ledger.json)、[缺失独立授权回执](../records/p3c/independent_authorization_receipt.json)。
- 执行代码：tools/p3c_runtime.py、p3c_supervise.py、p3c_acceptance.py、p3c_analyze.py。PI工具：p3c_sign_authorization.py、p3c_run_authorized.py。
- 阶段状态WAITING_INDEPENDENT_PI_SIGNATURE；没有启动后台测试或重新启用定时监控。P3-B模型和工作路径不变，论文定稿及扩展实验未授权。
'''
    (REPO/'reports/P3_C_启动准备与待签署回执.md').write_text(document,encoding='utf8',newline='\n')
    print(json.dumps({'CPU_tests':cpu['tests_run'],'new_GPU_seconds':0,'status':'WAITING_INDEPENDENT_PI_SIGNATURE',
        'request_sha256':sha(stage/'authorization_request.json')},ensure_ascii=False))

if __name__=='__main__':main()
