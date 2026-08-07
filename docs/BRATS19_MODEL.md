# Dataset002_BRATS19 nnU-Net v2 权重

## 本机安装位置

原始模型包：

`E:\Dataset002_BRATS19.zip`

SHA-256：

`F45EFF0624604333A80CA5E6EA9C71153D908A4C2082DF9038051DD7C4B2491F`

已安装到项目的 nnU-Net results：

`runtime/nnunet/nnUNet_results/Dataset002_BRATS19/nnUNetTrainer__nnUNetPlans__3d_fullres`

该目录属于运行产物，已被`.gitignore`排除，不会进入源码仓库或源码压缩包。

## 已验证配置

- Dataset ID：`2`
- Dataset：`Dataset002_BRATS19`
- Trainer：`nnUNetTrainer`
- Plans：`nnUNetPlans`
- Configuration：`3d_fullres`
- Folds：`0, 1, 2, 3, 4`
- Checkpoint：`checkpoint_final.pth`
- MRI通道：T1、T1ce、T2、FLAIR
- 训练结束epoch：`1001`
- 分割输出头：5类

五个fold、`plans.json`和`dataset.json`均已检查存在。

已使用`nnunetv2 2.8.1`在CPU上成功初始化fold 0：

- 网络：`PlainConvUNet`
- 参数量：`30,791,001`
- segmentation heads：`5`

模型使用旧版nnU-Net v2 plans格式。2.8.1会给出旧格式提示并自动重建架构参数；
本模型已完成重建和checkpoint加载验证。

## 重要：自定义标签映射

这个模型不是项目默认的BraTS21连续标签模型。其`dataset.json`名称定义为：

- `0`：background
- `1`：edema（元数据名称与checkpoint实际数值语义不一致）
- `2`：nonenhancing（元数据名称与checkpoint实际数值语义不一致）
- `3`：empty
- `4`：enhancing

使用与原始输入SHA-256严格匹配的`BraTS2021_00495`真值逐体素复核后确认，
checkpoint实际保留标准BraTS数值语义：标签1对应nonenhancing，标签2对应edema。
项目通过`brats19_preserved`输出标签配置保留1、2、4，仅将空类别3转为背景：

- 模型`0 → BraTS 0`
- 模型`1 → BraTS 1`
- 模型`2 → BraTS 2`
- 模型`3 → BraTS 0`
- 模型`4 → BraTS 4`

不能只依据`dataset.json`中的类别名称交换标签1和2；否则WT和ET不变，但TC及
水肿量化会被错误计算。也不能使用默认的`standard_nnunet`映射处理该模型，
因为默认配置会把内部标签3解释为增强区域。

fold 0单病例修正后独立Dice：WT `0.965846`、TC `0.978066`、ET `0.947714`。

本机`.env`已配置：

```dotenv
NNUNET_ROOT=./runtime/nnunet
BTA_NNUNET_DATASET_ID=2
BTA_NNUNET_CONFIGURATION=3d_fullres
BTA_NNUNET_PLANS=nnUNetPlans
BTA_NNUNET_TRAINER=nnUNetTrainer
BTA_NNUNET_FOLDS=["0","1","2","3","4"]
BTA_NNUNET_CHECKPOINT=checkpoint_final.pth
BTA_NNUNET_OUTPUT_LABEL_PROFILE=brats19_preserved
```

## 批量推理命令

```powershell
python -m segmentation.inference `
  --input-dir "D:\BraTS\Input" `
  --output-dir "D:\BraTS\Predictions" `
  --nnunet-root ".\runtime\nnunet" `
  --dataset-id 2 `
  --configuration 3d_fullres `
  --plans nnUNetPlans `
  --trainer nnUNetTrainer `
  --folds 0 1 2 3 4 `
  --output-label-profile brats19_preserved `
  --gpu-id 0
```

推理完成后：

- `nnunet_predictions`保存模型原始五类输出。
- `brats_predictions`保存统一后的标准BraTS `0/1/2/4` mask。

## GPU验证与适用范围

2026-08-07已在Tesla T4使用fold 0至4集成，对10例UPENN-GBM独立机构队列专家标签
完成验证，宏平均WT `0.895287`、TC `0.918673`、ET `0.862139`。逐病例指标、
GPU环境元数据和实验口径见公开验证记录：

[`validation/upenn-10/`](validation/upenn-10/README.md)

这是外部提供的BraTS19自定义模型包，仍缺原始训练代码版本、完整训练数据清单和
临床审核记录。UPENN结果属于独立机构的10例小样本验证，不能代表大规模临床泛化，
也不得直接作为临床诊断依据。
