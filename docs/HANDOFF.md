# BrainTumor-Agent 工作交接

更新时间：2026-07-29

## 一、当前已完成

### 1. 工程模块

- BraTS四模态NIfTI读取、几何检查、归一化和processed数据保存。
- Matplotlib静态四窗与Plotly放射科风格交互阅片器。
- nnU-Net v2数据转换、训练、推理、五折集成和Dice评估脚本。
- WT、TC、ET、水肿体积、最大径、区域比例和位置分析。
- FastAPI上传、分析编排、结果接口基础实现。
- Qwen、RAG、报告等工程接口和基础结构。

### 2. Dataset002_BRATS19权重接入

原始文件：

`E:\Dataset002_BRATS19.zip`

SHA-256：

`F45EFF0624604333A80CA5E6EA9C71153D908A4C2082DF9038051DD7C4B2491F`

安装位置：

`runtime/nnunet/nnUNet_results/Dataset002_BRATS19/nnUNetTrainer__nnUNetPlans__3d_fullres`

已确认：

- Dataset ID：2
- Trainer：`nnUNetTrainer`
- Plans：`nnUNetPlans`
- Configuration：`3d_fullres`
- Folds：0、1、2、3、4
- 五个`checkpoint_final.pth`均存在
- fold 0成功加载为`PlainConvUNet`
- 参数量：30,791,001
- segmentation heads：5
- checkpoint epoch：1001
- nnU-Net v2 2.8.1可自动重建旧版plans并加载权重

详细说明见`docs/BRATS19_MODEL.md`。

### 3. 自定义标签兼容

该模型不是标准的项目BraTS21连续标签模型。模型输出：

- 0：background
- 1：edema
- 2：nonenhancing
- 3：empty
- 4：enhancing

已增加`brats19_preserved`配置，统一转换为标准BraTS：

- 0 → 0
- 1 → 2
- 2 → 1
- 3 → 0
- 4 → 4

相关代码：

- `segmentation/prepare_dataset.py`
- `segmentation/inference.py`
- `backend/app/core/config.py`
- `backend/app/services/analysis.py`
- `backend/app/services/dependencies.py`

本机`.env`已切换为Dataset002和`brats19_preserved`。

### 4. Python环境

- 已安装`nnunetv2 2.8.1`。
- 非PyTorch依赖已安装。
- `pip check`结果：`No broken requirements found`。
- 当前PyTorch：`2.12.1+cpu`。
- 当前没有CUDA，尚未执行真实病例推理。

## 二、验证状态

已通过：

- 全工程测试：62 passed。
- 全仓Ruff检查：通过。
- 新增BraTS19标签转换测试。
- BraTS19预测目录恢复测试。
- segmentation + backend定向测试：17 passed。
- 本次修改文件Ruff检查通过。
- fold 0模型架构和checkpoint实际加载成功。
- Dataset002五折推理dry-run命令生成正确。

未完成：

- 没有真实四模态病例，尚未做端到端推理。
- 没有对应真值，尚未验证该权重的WT/TC/ET Dice。
- 没有验证五折集成后的标签3“empty”实际出现比例。

## 三、下一步优先级

### P0：GPU环境验证

在NVIDIA GPU机器安装与驱动匹配的CUDA PyTorch，确认：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

不要直接沿用当前CPU PyTorch做正式推理。

### P0：单病例fold 0冒烟推理

准备一个包含T1、T1ce、T2、FLAIR的BraTS病例，先只跑fold 0：

```powershell
python -m segmentation.inference `
  --input-dir "D:\BraTS\Input" `
  --output-dir "D:\BraTS\SmokePrediction" `
  --nnunet-root ".\runtime\nnunet" `
  --dataset-id 2 `
  --configuration 3d_fullres `
  --plans nnUNetPlans `
  --trainer nnUNetTrainer `
  --folds 0 `
  --output-label-profile brats19_preserved `
  --gpu-id 0
```

检查：

- 是否成功生成mask。
- 模型原始输出标签是否仅为0–4。
- 标签3占比是否异常。
- 恢复后标签是否仅为0、1、2、4。
- mask与MRI的shape、spacing和affine是否一致。

### P1：五折正式推理

单折通过后改为：

```powershell
--folds 0 1 2 3 4
```

若显存不足增加：

```powershell
--not-on-device
```

### P1：量化和阅片

```powershell
python -m feature_extract.tumor_measure `
  --mask "D:\BraTS\SmokePrediction\brats_predictions\CASE.nii.gz" `
  --output "D:\BraTS\SmokePrediction\CASE_measurement.json"
```

```powershell
python -m data_process.visualize `
  --input-dir "D:\BraTS\Input\CASE" `
  --seg-path "D:\BraTS\SmokePrediction\brats_predictions\CASE.nii.gz" `
  --label-space brats `
  --output-dir "D:\BraTS\Viewer"
```

### P1：独立Dice评估

```powershell
python -m segmentation.evaluate `
  --ground-truth-dir "D:\BraTS\GroundTruth" `
  --prediction-dir "D:\BraTS\Predictions\brats_predictions"
```

必须记录WT、TC、ET Dice后，才能判断该外部模型是否适合项目。

### P2：后端端到端

- 启动FastAPI。
- 上传四模态病例。
- 调用`POST /api/v1/analyze`。
- 核对mask下载、量化JSON、阅片器和报告输入。
- 增加GPU任务队列/并发锁的压力测试。

### P2：最终工程验收

```powershell
pytest -q -p no:asyncio -p no:cacheprovider
```

```powershell
ruff check .
```

源码压缩包需要在代码最终确认后重新生成。不要把`.env`、NIfTI病例、运行产物或
模型权重提交到GitHub。

## 四、风险提醒

- 模型名是BRATS19，但`dataset.json`包含大量BraTS2021样式病例名和80,064条训练项，
  训练数据组织明显经过自定义处理。
- `empty=3`是非标准类别，必须保留专用标签转换，不能改回默认映射。
- 模型使用旧版plans；虽然2.8.1加载成功，升级nnU-Net后需重新验证。
- 尚无来源说明、训练代码版本、独立验证指标和临床审核记录。
- 仅用于科研验证，不得直接用于临床诊断。
