# P3-C：PI 独立签署与启动

当前只有工程准备通过，没有真实签名授权，GPU合成入口与官方测试均未执行。以下步骤由PI或PI控制的独立签发流程完成，执行代理未代签。

1. 审阅 `records/p3c/authorization_request.json`，核对本次冻结400项范围、3 GPU-hours子上限、160 GPU-hours总上限、四份实际执行程序SHA、CPU验收SHA、执行顺序SHA和P3-B seal。
2. 在PI控制的独立终端运行签署辅助工具（可以用已有独立签发服务替代）。工具只接收PI已有的至少32字节HMAC密钥，隐藏输入，不生成或保存密钥。示例：

```powershell
python tools/p3c_sign_authorization.py --request records/p3c/authorization_request.json --output E:/Server_file/tl_cls_v1/test_authorization.json --pi-independent-review
```

这条命令只产生签名JSON，不会启动测试；`--pi-independent-review` 表示PI实际审阅，不能由训练执行代理自行勾选。HMAC不提供人类身份认证，独立终端/独立流程由PI控制。本项目不因这个字段自动宣称运维权限分离已被密码学证明。

3. 将签发的JSON按现有服务器文件规则中转至原工作目录下 `records/p3/test_authorization.json`。文件很小；传输后核验该文件大小/SHA，成功后只清理对应中转文件。不要上传HMAC密钥，不要提交或发送真实密钥到聊天、仓库、日志和命令行。
4. 由PI在受控服务器终端运行下列启动器，隐藏输入同一HMAC密钥。密钥只在进程运行时环境中传递，不落盘；启动器立即返回后台PID，无需前台持续监测。

```bash
cd /root/user_data/tl_cls_v1
/root/miniconda3/envs/tl_cls_v1/bin/python tools/p3c_run_authorized.py
```

后台顺序为：严格验证签名和全部封存身份 → 两卡纯合成GPU入口验收并计费 → 通过后执行固定400项 → 独立预测重算/统计/封存。任一门失败立即停止，不能临时更换src源码或伪造code_commit。新增GPU验收和正式评价共同受3小时子上限和原160小时总上限约束。

若使用已有外部PI签发系统，应输出同样的 `P3_C_authorization_v1` 载荷与HMAC，并通过受控启动环境注入 `TLCLS_TEST_APPROVAL_HMAC_KEY`；不必采用本辅助工具。

训练seal、80个summary、原模型和P2选参保持冻结。新的共享GPU账本内容会追加，旧seal中的P3-B账本SHA仍对应保留的 `records/p3c/p3b_gpu_ledger.sealed.json`；追加费用另有签名链，不重写旧seal。正式预测保留在服务器，公开仓库只回交指标、统计、图和审核摘要。

发生技术中断时，只允许在保全证据、旧attempt全部结算、原因与签名仍有效后显式使用 `tools/p3c_supervise.py --resume`。预算/存储/未知状态阻塞需先回交审核，不能把resume当作无条件重启。半成品不视为完成，已完成预测在恢复时验证SHA后才可跳过；不会根据准确率安排重跑。
