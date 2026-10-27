# P3-C PI直接授权与后台启动回执

状态：**已获PI直接授权并后台启动，尚未完成400项交付。** 本轮取消独立签发，不再等待外部密钥或重复确认。

- [PI直接授权补充](../docs/P3_C_PI直接授权补充_20261027.md)、[服务器授权记录](../records/p3c/PI_direct_approval.json)、[启动机器回执](../records/p3c/direct_PI_launch.json)。
- 新授权模式为 `direct_user_PI_authorization`，独立签发声明为false，豁免标记为true。保留原HMAC严格准入，只将签发流程交由本轮授权的代理管理。
- 更新后28项CPU合成回归通过；四份实际运维源码SHA与CPU回执、测试授权绑定。双卡合成入口已完成并自动进入正式评价阶段（已观测两个p3c_evaluation attempt）。正式400项尚未完成，不提前报告test成绩。
- core/src执行提交仍700a1adc，训练科学基线、P3-B seal、80个summary、模型/权重/数据划分及16组LR不变。运维tools源码独立按SHA记录，归档提交与core提交分开。
- 冻结顺序80 best → 80 last → 240 prefix，类内run_id/fraction稳定排序；原预测与checkpoint留服务器，不按test结果重选。
- 子硬上限3 GPU-hours；P3共享总上限160 GPU-hours，含先前124.610609507小时，旧92项attempt不扣减。Git中的准备期零费用子账本是历史快照，当前费用以服务器及带时间的launch_budget_snapshot为准。
- 密钥仅在服务器项目外的运行时凭据存储（0700目录/0600文件）与进程环境，未下载、未写脚本字面量/命令行/日志/Git。已有授权文件保留于服务器authorization_history，本轮凭证另行记录直接授权事实。
- 服务器独立30分钟只读检查已启动，桌面tlcls-p3c-30min已配置ACTIVE并读回确认；正常保持安静。完成且所有guardian结算、独立audit通过后回收文本、分析作图并推送；不依赖前台watch。

最近成功状态快照（Unix 1793104854.2565205）：已计新增GPU 0.178716/3小时，总124.789325/160小时；该快照当时还在GPU合成入口，之后已观测正式评价attempt。它不代表当前实时费用。SSH出现间歇性协议握手EOF，连接失败与测试失败分开记录；后台守护与预算独立运行，未因SSH断连重复启动。后续回收由定时检查继续处理。

回收程序：ops/collect_p3c.py（默认等待后台退出和全部attempt结算；--launch-only只取启动小型证据）。排除密钥引用、NPZ预测和模型。公开仓库只交付汇总、图、哈希与审核材料。尚未生成本轮P3-C最终验收；未完成必须明确报告缺失ID和实际成本。

启动证据补齐：已回收并核验双卡GPU验收机器收据，状态passed，合计0.188573 GPU-hours（包括原严格准入及加载），全部计入3小时。当前已归档状态running、两个正式p3c_evaluation attempt；启动归档只有11份文本，约44KB压缩包，明确NOT_FINAL。见[启动审计](../records/p3c/startup_audit.json)与[启动原始证据](../evidence/P3C_start_20261027)。

共享GPU账本已按P3-C正常追加，因此它的当前SHA不再等于旧P3-B seal字段；旧字段仍严格对应records/p3c/p3b_gpu_ledger.sealed.json。原92项attempt逐项相等，未改旧seal来迎合新账本。启动快照中两项正式attempt尚未结算，不把部分费用冒充最终费用。

公开启动证据为已核验原始快照的10文件子集，明确排除完整运行授权；完整授权保留服务器与受控本地证据，密钥始终只在服务器。公开的子集清单保留原始快照/归档SHA，未改写原始私人归档。
