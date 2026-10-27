# P3-C 官方测试结果

状态：passed；完成400/400；新增2.072720/3 GPU-hours；P3总126.683330/160。

主指标为预先在验证集选择的best checkpoint的官方test Top-1。没有使用test重新选参或选模。原始预测和混淆矩阵留服务器，完整逐角色指标及逐seed数据见CSV。

|配置（主best）|n/5|Test Top-1均值 ± 样本SD (%)|
|---|---:|---:|
|cifar100-r50-adapter|5|82.934 ± 0.348|
|cifar100-r50-full_ft|5|84.484 ± 0.480|
|cifar100-r50-linear_probe|5|74.396 ± 0.237|
|cifar100-r50-scratch|5|72.272 ± 0.207|
|cifar100-vitb16-adapter|5|93.588 ± 0.110|
|cifar100-vitb16-full_ft|5|93.688 ± 0.099|
|cifar100-vitb16-linear_probe|5|87.948 ± 0.162|
|cifar100-vitb16-scratch|5|49.788 ± 0.837|
|svhn-r50-adapter|5|96.574 ± 0.131|
|svhn-r50-full_ft|5|96.273 ± 0.104|
|svhn-r50-linear_probe|5|70.601 ± 0.085|
|svhn-r50-scratch|5|96.742 ± 0.117|
|svhn-vitb16-adapter|5|97.469 ± 0.084|
|svhn-vitb16-full_ft|5|97.993 ± 0.111|
|svhn-vitb16-linear_probe|5|67.407 ± 0.060|
|svhn-vitb16-scratch|5|91.633 ± 0.376|

两项预注册比较（其余均探索性）：

- {"comparison": "cifar100-vitb16-linear_probe-minus-full_ft", "n_required": 5, "n_complete_pairs": 5, "status": "complete", "mean_difference_pp": -5.740000000000004, "sample_SD_pp": 0.23043437243605974, "CI95_low_pp": -6.0261220642025375, "CI95_high_pp": -5.45387793579747, "paired_t_two_sided_p": 6.22042487495037e-07, "Holm_adjusted_p": 1.244084974990074e-06, "superiority_Holm_alpha_0_05": false, "noninferiority_margin_pp": -0.5, "noninferiority_CI95_lower_rule": false}
- {"comparison": "svhn-r50-adapter-minus-full_ft", "n_required": 5, "n_complete_pairs": 5, "status": "complete", "mean_difference_pp": 0.3011677934849445, "sample_SD_pp": 0.1870028872738924, "CI95_low_pp": 0.068973040796795, "CI95_high_pp": 0.533362546173094, "paired_t_two_sided_p": 0.022734182136255474, "Holm_adjusted_p": 0.022734182136255474, "superiority_Holm_alpha_0_05": true, "noninferiority_margin_pp": null, "noninferiority_CI95_lower_rule": null}

双侧配对t检验仅用于两项固定比较，按Holm校正；非劣性只对CIFAR/ViT LP−FT用95%双侧区间下界严格大于−0.5pp判定。预算前缀属于同轨迹重复测量，不能当作独立样本。缺少任一配对seed时不作完整主推断。
