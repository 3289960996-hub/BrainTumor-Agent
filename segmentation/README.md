# BraTS 2021 nnU-Net V2分割

本模块完成BraTS 2021数据转换、nnU-Net V2规划与预处理、GPU训练、推理标签还原和Dice评估。

## 标签约定

BraTS原始标签不是连续整数，因此训练前使用nnU-Net官方转换约定：

- BraTS背景0 → nnU-Net 0
- BraTS水肿ED 2 → nnU-Net 1
- BraTS坏死/非强化核心NCR/NET 1 → nnU-Net 2
- BraTS强化肿瘤ET 4 → nnU-Net 3

区域训练定义：

- Whole Tumor：1、2、3
- Tumor Core：2、3
- Enhancing Tumor：3

推理后默认还原为BraTS 0、1、2、4标签。

## 1. 环境要求

建议：

- Linux或WSL2
- Python 3.11
- NVIDIA GPU和CUDA
- 9–11GB显存使用默认ResEnc M
- 24GB显存可使用ResEnc L

先根据CUDA版本安装兼容的PyTorch，再安装项目依赖：

`pip install -r requirements.txt`

四个脚本通过`--nnunet-root`自动配置：

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

也可以设置`NNUNET_ROOT`环境变量作为默认根目录。

## 2. 数据转换、规划和预处理

BraTS训练根目录应包含病例子目录，每个病例包含T1、T1ce、T2、FLAIR和seg。

`python -m segmentation.prepare_dataset --brats-dir D:\BraTS2021\TrainingData --nnunet-root D:\nnUNet_workspace --dataset-id 137 --plan-and-preprocess`

默认配置：

- 数据集：`Dataset137_BraTS2021`
- 配置：`3d_fullres`
- 规划器：`nnUNetPlannerResEncM`
- 计划：`nnUNetResEncUNetMPlans`

只检查将要运行的规划命令，可增加`--dry-run`。注意数据转换本身仍会执行。

24GB GPU使用ResEnc L时：

`python -m segmentation.prepare_dataset --brats-dir D:\BraTS2021\TrainingData --nnunet-root D:\nnUNet_workspace --planner nnUNetPlannerResEncL --plan-and-preprocess`

后续训练和推理应同时使用`--plans nnUNetResEncUNetLPlans`。

## 3. GPU训练

训练单个fold：

`python -m segmentation.train --nnunet-root D:\nnUNet_workspace --dataset-id 137 --folds 0 --gpu-ids 0 --npz`

顺序训练五折：

`python -m segmentation.train --nnunet-root D:\nnUNet_workspace --dataset-id 137 --folds 0 1 2 3 4 --gpu-ids 0 --npz`

断点续训：

`python -m segmentation.train --nnunet-root D:\nnUNet_workspace --folds 0 --gpu-ids 0 --continue-training`

训练完成后重新执行验证：

`python -m segmentation.train --nnunet-root D:\nnUNet_workspace --folds 0 --gpu-ids 0 --validation-only --npz`

多GPU DDP可指定`--gpu-ids 0,1 --num-gpus 2`。nnU-Net官方通常更推荐每张GPU独立训练一个fold，因为小网络的数据并行扩展效率有限。

## 4. 推理

默认使用五折集成、CUDA、测试时增强，并生成BraTS标签：

`python -m segmentation.inference --input-dir D:\BraTS2021\ValidationData --output-dir D:\BraTS_predictions --nnunet-root D:\nnUNet_workspace --gpu-id 0`

输出：

- `nnunet_input/`：标准四通道推理输入
- `nnunet_predictions/`：nnU-Net内部0、1、2、3标签
- `brats_predictions/`：还原后的BraTS 0、1、2、4标签

显存不足时可以增加`--not-on-device`；只训练了单折时使用`--folds 0`。

## 5. Dice验证

评估还原后的BraTS预测：

`python -m segmentation.evaluate --ground-truth-dir D:\BraTS2021\TrainingData --prediction-dir D:\BraTS_predictions\brats_predictions`

输出：

- Whole Tumor Dice
- Tumor Core Dice
- Enhancing Tumor Dice
- `evaluation.json`
- `evaluation_cases.csv`

评估nnU-Net内部标签时，增加：

`--ground-truth-label-space nnunet --prediction-label-space nnunet`

空区域约定：如果预测和真值中的某区域都为空，该区域Dice记为1。

## 6. 测试与命令检查

测试不启动训练，也不要求GPU：

`pytest segmentation/tests -q`

训练命令检查：

`python -m segmentation.train --nnunet-root D:\nnUNet_workspace --folds 0 --dry-run`

推理命令检查：

`python -m segmentation.inference --input-dir D:\BraTS2021\ValidationData --output-dir D:\BraTS_predictions --nnunet-root D:\nnUNet_workspace --dry-run`

## 7. 已接入的Dataset002_BRATS19权重

本机已将`E:\Dataset002_BRATS19.zip`安装到
`runtime/nnunet/nnUNet_results/Dataset002_BRATS19`。该模型使用
`nnUNetTrainer + nnUNetPlans + 3d_fullres`和五折权重。

它是自定义五类模型，推理时必须设置：

`--dataset-id 2 --plans nnUNetPlans --output-label-profile brats19_preserved`

完整模型身份、标签映射、推理命令和限制见`docs/BRATS19_MODEL.md`。
