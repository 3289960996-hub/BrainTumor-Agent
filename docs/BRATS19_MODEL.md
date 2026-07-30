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

这个模型不是项目默认的BraTS21连续标签模型。其`dataset.json`定义：

- `0`：background
- `1`：edema
- `2`：nonenhancing
- `3`：empty
- `4`：enhancing

项目通过`brats19_preserved`输出标签配置将其转换为标准BraTS标签：

- 模型`0 → BraTS 0`
- 模型`1 → BraTS 2`
- 模型`2 → BraTS 1`
- 模型`3 → BraTS 0`
- 模型`4 → BraTS 4`

不能使用默认的`standard_nnunet`映射处理该模型，否则增强区域会被错误解释。

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

## 适用范围与限制

这是外部提供的BraTS19自定义模型包，项目中没有对应的独立验证集指标、训练代码
版本或临床审核记录。目前只完成了包结构、checkpoint安全加载、网络输出头和标签
契约验证。本机当前PyTorch为CPU构建，没有执行真实病例推理。投入科研评估前应在
CUDA环境中对独立BraTS数据重新计算WT、TC、ET Dice；不得直接作为临床诊断依据。
