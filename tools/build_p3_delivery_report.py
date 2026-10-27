"""从独立验收 CSV 与封存台账生成阶段汇报和待审预算。"""
import csv
import datetime
import json
from pathlib import Path
import statistics
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def read(n):
    return json.loads((ROOT / n).read_text(encoding='utf8'))

def rows(n):
    return list(csv.DictReader((ROOT / 'reports' / n).open(encoding='utf8')))

def write(n, value):
    (ROOT / n).write_text(value, encoding='utf8', newline='\n')

def csv_write(n, values):
    with (ROOT / 'reports' / n).open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n'); w.writeheader(); w.writerows(values)

def main():
    audit = read('records/p3/delivery_audit.json')
    remote = read('records/p3/server_closeout_verification.json')
    assert audit['status'] == remote['status'] == 'passed'
    assert audit['gpu_ledger_sha256'] == remote['gpu_ledger_sha256']
    configs = rows('P3_validation_configurations.csv'); curves = rows('P3_epoch_curves.csv')
    runs = rows('P3_validation_runs.csv'); paired = rows('P3_primary_validation_differences.csv')
    ledger = read('records/p3/gpu_ledger.json'); attempts = list(ledger['attempts'].values())
    storage = read('records/p3/storage_latest.json'); seal = read('records/P3_sealed.json')
    csv_write('P3_attempts.csv', [{k: a.get(k) for k in ['attempt_id', 'run_id', 'kind', 'device',
        'execution_commit', 'started_unix', 'started_monotonic', 'ended_monotonic', 'charged_gpu_seconds',
        'returncode', 'termination_reason', 'status']} for a in attempts])
    samples = [json.loads(x) for x in (ROOT / 'records/p3/storage_ledger.jsonl').read_text().splitlines()]
    csv_write('P3_storage_timeline.csv', [{k: a.get(k) for k in ['unix', 'combined_bytes',
        'project_bytes', 'environment_non_symlink_bytes', 'filesystem_free_bytes',
        'artificial_storage_ceiling_enforced']} for a in samples])
    used = audit['GPU_hours']; remaining = 160 - used
    candidate = sum(a['charged_gpu_seconds'] for a in attempts if a['kind'] == 'candidate') / 3600
    start = min(a['started_unix'] for a in attempts)
    end = max(a['started_unix'] + a['charged_gpu_seconds'] for a in attempts)
    medians = {c['configuration_id']: statistics.median(float(r['validation_seconds']) for r in curves
        if r['configuration_id'] == c['configuration_id'] and int(r['epoch']) >= 2) for c in configs}
    inference = sum(t * 25 * (2 if k.startswith('cifar100') else 26032 / 7326) for k, t in medians.items()) / 3600
    proposal = {'schema': 'P3_C_review_proposal_v1', 'status': 'pending_user_review_NOT_AUTHORIZED',
        'P3_C_authorized': False, 'centralized_test_authorized': False,
        'seal_sha256': audit['seal_sha256'], 'evaluation_manifest_sha256': seal['evaluation_manifest_sha256'],
        'evaluation_roles': 400, 'distinct_checkpoint_prediction_keys': 400,
        'test_sample_counts_for_cost_scaling_only': {'cifar100': 10000, 'svhn': 26032},
        'P3_validation_seconds_median_epochs_2_to_50': medians,
        'scaled_inference_only_GPU_hours': inference, 'with_25_percent_future_margin_GPU_hours': inference * 1.25,
        'original_held_test_reserve_GPU_hours': 1.4646375681192052,
        'recommended_new_evaluation_subbudget_GPU_hours_PENDING_APPROVAL': 3.0,
        'recommended_budget_includes': ['checkpoint load and hash', 'inference', 'prediction serialization',
            'bounded validation timing and necessary diagnostics/retries'],
        'P3_used_GPU_hours': used, 'P3_total_cap_GPU_hours': 160,
        'projected_total_if_new_3h_subbudget_approved': used + 3,
        'additional_output_storage_planning_GiB': 5,
        'storage_is_incremental_plan_not_hardware_cap': True,
        'checkpoint_download': False, 'monetary_price': 'school server; no hourly price',
        'required_before_execution': ['explicit separate user approval of centralized test and subbudget',
            'freeze exact seal and evaluation manifest SHA', 'external HMAC signing secret; never stored in repository',
            'confirm unchanged 80-run identity, checkpoint hashes and evaluator identity',
            'all evaluation attempts charged to shared P3 ledger; reserve and exit guards remain'],
        'statistics_after_test': ['paired five-seed t intervals for two preregistered contrasts',
            'Holm correction for two preregistered superiority tests',
            'CIFAR-100 ViT LP-minus-FT noninferiority margin -0.5 percentage points',
            'retain all five seeds; no test-driven retuning or checkpoint substitution']}
    write('reports/P3_C_待审预算.json', json.dumps(proposal, ensure_ascii=False, indent=2) + '\n')
    p = '# P3-B 完整结果汇报与验收回执\n\n'
    p += '2026-10-27。**P3-B：PASS，80/80 正式运行完成并封存；P3-C：NO_GO，等待另行审核。** 所有结果均为训练/验证证据，没有官方测试预测或测试指标。\n\n'
    p += f'本轮从服务器回收 17,121 个小型文本证据文件，原始合计 41,053,549 bytes，压缩包 9,724,237 bytes（约 9.72 MB），逐文件大小/SHA-256 全部通过。508 个 P3 checkpoint 留服务器，没有回收大模型。\n\n'
    p += '## 完成状态与证据\n\n'
    p += '|项目|核验结果|\n|---|---|\n'
    facts = [('正式矩阵','16 配置 × seeds 17/42/2026/3407/65537 = 80；每次 50 epochs'),
        ('已提交训练','4,000 epochs；221,862,000 样本次；1,736,000 optimizer updates'),
        ('已提交验证','24,652,000 样本次；仅原冻结 validation split'),
        ('恢复与初始条件','80 个首次 attempt 均 fresh；4 次合法基础设施恢复，未继承 P2 checkpoint'),
        ('最佳模型','逐轮重算 Top-1、并列最早轮；best/last 指针及封存哈希吻合'),
        ('确定性配对','20 个 dataset/backbone/seed 初始头一致；1,000 个对应 epoch 四策略 sampler 顺序一致'),
        ('预算切片','240/240 available；每点选 deadline 之前最后一个完整、哈希验真的 checkpoint'),
        ('评价清单','80 best + 80 last + 240 prefix = 400 个唯一 evaluation_id；400 个不同缓存键'),
        ('实际执行来源','2,740 epochs 为 7247c661；1,260 epochs 为 700a1adc；54/26 个终态 summary 分别绑定两提交'),
        ('历史保全','两次恢复前54/65个完整run的保留哈希均一致；P1 40、P2 128个checkpoint重新计算哈希通过'),
        ('异常终态','数值失败/OOM/协议失效均为0；4个被暂停attempt保留并结算'),
        ('测试边界','P3_C_authorized=false，centralized_test_authorized=false；未构造测试 evaluator')]
    p += ''.join(f'|{a}|{b}|\n' for a,b in facts)
    p += '\n训练样本次数指完整提交事务的计数，**所有 attempt 的实际物理样本处理总数 UNKNOWN**：中断时未提交的半轮工作没有完整采样日志，不能把 221,862,000 冒充含失败重算的精确物理总量。GPU 台账则完整包含这些时间。四个恢复 run 的 summary `fresh_initialization=false` 表示“最后一次 attempt 从同 run checkpoint 恢复”；首次 attempt 的 `resume=false`、`resumed_generation=null` 已逐一验证，不应误读为继承 P2。\n\n'
    p += '\n`progress.json` 是调度循环最后快照，可能仍标记 running；终态以 stage_status=P3_runs_terminal、80个completed summary、92个finalized attempt及P3 seal共同确认，未将旧进度字段当作仍在训练。\n\n'
    p += '## 五种子最佳验证结果\n\n'
    p += '下表为每个 run 从 50 轮中选择的最佳验证 Top-1 的均值 ± 样本 SD，n=5；不是标准误或置信区间。选择使用了同一验证集，成绩具有选择偏差。`last` 的五 seed 均值/SD、80个原始成绩、4,000轮 train/validation loss及准确率见随附 CSV。\n\n'
    p += '|配置|冻结 LR|最佳验证 Top-1 (%)|最后轮验证均值 (%)|累计 GPU-hours|\n|---|---:|---:|---:|---:|\n'
    for c in configs:
        p += f"|{c['configuration_id']}|{float(c['learning_rate']):g}|{float(c['best_validation_mean']):.3f} ± {float(c['best_validation_SD']):.3f}|{float(c['last_validation_mean']):.3f}|{float(c['GPU_hours']):.4f}|\n"
    p += '\n![五种子验证成绩](../figures/P3/validation_results.png)\n\n'
    p += '## 两项预注册主比较的验证观察\n\n'
    p += '|比较（前者减后者）|五个配对差值，seed依次17/42/2026/3407/65537 (pp)|均值 ± 配对 SD (pp)|\n|---|---|---:|\n'
    for key in sorted({r['comparison'] for r in paired}):
        values = [float(next(x['validation_difference_pp'] for x in paired if x['comparison'] == key and int(x['seed']) == seed)) for seed in [17,42,2026,3407,65537]]
        p += f"|{key}|{', '.join(f'{v:+.3f}' for v in values)}|{statistics.mean(values):+.3f} ± {statistics.stdev(values):.3f}|\n"
    p += '\nCIFAR-100/ViT 的 LP 在五个 seed 上均低于 Full FT，均值差 −5.316 pp；当前验证证据不支持 LP 优势。SVHN/R50 的 Adapter 五个配对差值均为正，均值 +0.278 pp，但 Scratch 验证均值 96.134% 还高于 Adapter 95.820%；因此不能把局部主比较推广成 Adapter 对所有方法最优。\n\n'
    p += 'CIFAR-100/ViT 的 Adapter 与 Full FT 最佳验证均值相差 −0.032 pp，只能描述为数值接近，不能据此声称等效。SVHN 的冻结骨干 Linear Probe 较低（R50 69.255%、ViT 65.618%），与全量/Adapter 的差距跨五 seed 稳定；ViT Scratch 在 CIFAR-100 为49.100%。这些是当前冻结配方下的观察，尚无因果消融，不能归因到某一单独结构、LR 或数据域因素，也不能事后扩网格。\n\n'
    p += '![主比较的逐seed差值](../figures/P3/primary_validation_differences.png)\n\n'
    p += '## 学习过程与共同时间预算\n\n'
    p += '验证曲线保留50轮、不平滑。LP 较早接近平稳；Full FT 和 Adapter 在两组 SVHN 上达到较高验证成绩，Scratch/ViT 仍呈更慢的提升。68/80 个最佳轮次早于第50轮，故 best 与 last 的评价角色需要同时保留。没有观察到非有限更新终止，但这不能证明旧稿 NaN 的因果机制已解决。\n\n'
    p += '![逐轮验证曲线](../figures/P3/validation_curves.png)\n\n'
    p += '四组 B 仍取冻结 JSON 的完整精度，未依据成绩调整。时间包括自身初始化、训练、验证、保存/哈希/必要退出及恢复开销，排除排队和暂停空档。每个 prefix 是同一条训练轨迹的复用，不是额外独立训练；240 个都找到了预算前合法快照，逐一核对完成时刻与最晚合法轮次。原始快照和保护哈希均列在 prefix_manifest/checkpoint_inventory。\n\n'
    p += '![共同时间预算下的验证结果](../figures/P3/time_prefix_validation.png)\n\n'
    p += '## 暂停原因、恢复与执行身份\n\n'
    p += '1. 首次暂停由旧预测公式给已经发生的费用再乘25%余量引起，不是真实超过160小时。修正为“实际已耗时 + 1.25×未来训练及预留”，硬封顶和历史费用均保留。54个已完成run没有重跑；R50 Full FT seed42、Scratch seed2026从第37/3轮恢复。真实checkpoint下一步更新逐位一致的GPU验收及20项CPU验收已留档。\n'
    p += '2. 第二次暂停由 CIFAR-100/ViT/LP seed17 的完整第2–3轮55.2231s、相对冻结P2中位50.0932s慢10.2408%触发；仅比10%线多约0.1206s。与P2同第2–3轮比较时增幅约6.96%；主要观测增量来自checkpoint保存路径（约2.15s），训练约1.35s、验证约0.10s。证据支持固定snapshot/hash/GC开销解释，未采样历史磁盘带宽，不能断言物理磁盘故障。按既有继续授权完成范围限定调查，只接受该配置五个预注册seed的已知开销；其他配置和未知故障没有全局放行。两个LP从第3/35轮恢复。\n\n'
    p += '实际源码只在已审查的6个运维文件变化；common/data/models/engine、事务实现、测试门禁及原科学身份不变。54个summary仍保持7247c661，26个为700a1adc，不伪造全阶段同一执行提交；独立核验逐轮绑定真实Git blob，检查54/65份保留哈希。当前报告和证据归档commit是交付提交，与服务器实际执行700a1adc分开记录。\n\n'
    p += '## GPU 与存储结算\n\n'
    p += f'实际占用 **{used:.9f}/160 GPU-hours**，剩余 {remaining:.9f}。其中84个正式attempt共{candidate:.9f}小时，8个GPU工程验收attempt共{used-candidate:.9f}小时；4个被暂停attempt已计入，不能从总账扣减。两个GPU的所有区间逐卡检查无重叠。\n\n'
    p += '|attempt终止原因|数量|GPU-hours|\n|---|---:|---:|\n'
    for key in sorted({a['termination_reason'] for a in attempts}):
        subset = [a for a in attempts if a['termination_reason'] == key]
        p += f"|{key}|{len(subset)}|{sum(a['charged_gpu_seconds'] for a in subset)/3600:.9f}|\n"
    p += f'\n台账最早GPU验收至最后GPU任务退出：{datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat()} 至 {datetime.datetime.fromtimestamp(end, datetime.timezone.utc).isoformat()}，日历跨度 {(end-start)/3600:.3f} 小时；包含排队/长暂停，不能当作GPU小时。封存UTC为 {seal["utc"]}。学校服务器无计价，不虚构金额。\n\n'
    p += f'服务器为2×RTX3090 24GiB、16 vCPU、64GiB内存；训练日志的最大单run显存峰值allocated {max(float(x["peak_allocated_GiB"]) for x in runs):.3f} GiB、reserved {max(float(x["peak_reserved_GiB"]) for x in runs):.3f} GiB。封存时项目+独立环境逻辑文件占用 {storage["combined_bytes"]/2**30:.3f} GiB，物理盘可用约 {storage["filesystem_free_bytes"]/2**40:.3f} TiB。此统计排除symlink、每根目录内hardlink去重，不等于conda共享包缓存的独占物理空间。P3 checkpoint 508个，共{audit["checkpoint_bytes"]/2**30:.3f} GiB；P1/P2原始checkpoint保留。\n\n'
    p += '200GiB是历史最低资源保障，不再是人为存储封顶；遵循用户后续授权保留物理空间10GiB退出余量+4GiB原子保存保护。存储台账为分时采样，封存统计之后本次文本导出/复核收据仍会少量增加项目占用；不把台账中的历史剩余额度当作当前硬限制。\n\n'
    p += '## 下一阶段提案：仅待审，不执行\n\n'
    p += f'已冻结的评价草案含400个角色、400个不同checkpoint推理键，本次没有可省略的重复键。按P3第2–50轮验证时间中位数及已知test样本规模外推，纯推理约 **{inference:.3f} GPU-hours**，加25%未来余量约 **{inference*1.25:.3f}**；不同batch、模型加载、哈希与预测压缩写入尚未实测。建议另行批准 **3.0 GPU-hours 的集中评价子预算**（含加载、推理、写入、必要诊断/重试），而不是把原1.464638小时估计当作已测上限。它仍位于160总封顶内，批准后总量最多约127.611小时；本轮没有动用这项额度。预计新增输出按5GiB规划，保留全部checkpoint，无需回收大文件。\n\n'
    p += '启动仍需用户单独明确批准P3-C、锁定seal与评价manifest的SHA、外部HMAC签名授权和原共享台账保护。不得测试后更换LR或模型；正式统计按冻结两项配对五seed t区间、两项优势检验Holm校正及CIFAR/ViT LP−FT −0.5pp非劣界执行。本报告不对验证成绩做正式优势/非劣裁定，不启动扩展实验或论文定稿。\n\n'
    p += '## 交付索引与复核方式\n\n'
    p += '- [独立验收](../records/p3/delivery_audit.json)、[服务器只读复核](../records/p3/server_closeout_verification.json)、[P3封存](../records/P3_sealed.json)、[评价草案](../records/p3/evaluation_manifest.draft.json)。\n'
    p += '- [逐run验证结果](P3_validation_runs.csv)、[16组汇总](P3_validation_configurations.csv)、[全部逐轮曲线数据](P3_epoch_curves.csv)、[prefix验证数据](P3_prefix_validation.csv)、[逐attempt台账](P3_attempts.csv)、[存储时间线](P3_storage_timeline.csv)、[后续预算JSON](P3_C_待审预算.json)。\n'
    p += '- [完整原始文本归档](../evidence/P3_final_20261027/text_evidence.tar.gz)、[逐文件manifest](../evidence/P3_final_20261027/snapshot_manifest.json)、[传输收据](../evidence/P3_final_20261027/receipt.json)。17,121个文件逐一核验；P3二进制哈希由服务器独立终态审计提供，本机仅核对其封存及引用一致性。\n'
    p += '- 原始归档解压至独立目录后，在仓库运行 `python tools/verify_p3_delivery.py --snapshot <解压目录>` 可重算验收和CSV（缺省未导出的P2账本从已接受归档读取并核对冻结SHA）。图源为 `tools/plot_p3_results.py`；全部图提供PDF/SVG/PNG及对齐、字体、碰撞QA记录。\n\n'
    p += f'封存SHA-256：`{audit["seal_sha256"]}`。GPU账本SHA-256：`{audit["gpu_ledger_sha256"]}`。阶段终点是P3-B已封存；P3-C与集中测试继续关闭。\n'
    write('reports/P3_完整结果汇报与验收.md', p)
    print(json.dumps({'report': 'reports/P3_完整结果汇报与验收.md', 'GPU_hours': used,
        'P3_C_authorized': False, 'proposed_evaluation_subbudget_GPU_hours': 3.0}))

if __name__ == '__main__':
    main()
