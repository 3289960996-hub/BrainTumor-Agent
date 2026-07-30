# Google Colab GPU 训练说明

## 已选择的路线

本项目使用 Google Colab 免费 GPU 真实训练 BraTS2021 四模态 nnU-Net V2 模型。
没有把其他数据集、其他网络结构的公开权重冒充为本项目训练结果。

可直接上传并运行的 Notebook：

`notebooks/BrainTumor_Agent_nnUNetv2_Colab.ipynb`

## 为什么不直接替换成公开权重

目前可确认的可信公开模型之一是 MONAI 官方 `brats_mri_segmentation` bundle，
但它基于 BraTS 2018，网络和推理约定也不是本项目的 nnU-Net V2 Dataset137。
它可以作为单独标注的 Demo fallback，不能用于证明“已完成 BraTS2021 nnU-Net V2
训练”。因此当前主路线保留真实训练。

## 需要用户准备的内容

由于 BraTS2021 数据受数据使用协议约束，项目不能代替用户公开下载或重新分发。
需要从官方授权渠道取得训练集，并保证每个病例具有：

- `*_t1.nii.gz`
- `*_t1ce.nii.gz`
- `*_t2.nii.gz`
- `*_flair.nii.gz`
- `*_seg.nii.gz`

将病例目录压缩成 `BraTS2021_TrainingData.zip`。

Google Drive 建议布局：

```text
MyDrive/
└── BrainTumor-Agent/
    ├── BrainTumor-Agent-Colab.zip
    └── BraTS2021_TrainingData.zip
```

Notebook 运行后还会生成：

```text
MyDrive/
└── BrainTumor-Agent/
    ├── nnUNet_storage/
    │   ├── nnUNet_raw/
    │   ├── nnUNet_preprocessed/
    │   └── nnUNet_results/
    └── exported_models/
        └── Dataset137_BraTS2021_<trainer>_fold0.zip
```

## 两种训练配置

### 实习项目演示

Notebook 默认值：

```python
TRAINING_PROFILE = "demo_100epochs"
FOLD = "0"
```

实际 trainer 为 `nnUNetTrainer_100epochs`。这是用真实完整数据训练的模型，
能够形成可验证 checkpoint、预测和 Dice，但不是 nnU-Net 标准的完整训练日程。

后端 `.env` 应对应设置：

```dotenv
BTA_NNUNET_TRAINER=nnUNetTrainer_100epochs
BTA_NNUNET_FOLDS=["0"]
BTA_NNUNET_PLANS=nnUNetResEncUNetMPlans
BTA_NNUNET_CONFIGURATION=3d_fullres
BTA_NNUNET_CHECKPOINT=checkpoint_final.pth
```

### 标准完整实验

Notebook 改为：

```python
TRAINING_PROFILE = "full_1000epochs"
FOLD = "0"
```

fold 0 完成后依次改为 1、2、3、4。此时 trainer 为标准
`nnUNetTrainer`。五个 fold 都完成后，后端配置为：

```dotenv
BTA_NNUNET_TRAINER=nnUNetTrainer
BTA_NNUNET_FOLDS=["0","1","2","3","4"]
```

## 断点续训

nnU-Net 每 50 epochs 写入 `checkpoint_latest.pth`。Notebook 将
`nnUNet_results` 直接持久化到 Google Drive。Colab 断线后：

1. 重新打开 Notebook 并选择 GPU；
2. 保持同一个 `TRAINING_PROFILE` 和 `FOLD`；
3. 从第一格重新顺序运行；
4. 训练格发现 `checkpoint_latest.pth` 后自动传入
   `--continue-training`。

不要删除 `nnUNet_storage/nnUNet_results` 中对应的 fold 目录。

## 将模型导回 Windows 项目

训练完成后，从 Drive 的 `exported_models` 下载模型 zip，解压到本机设置的
`NNUNET_ROOT/nnUNet_results`。最终应形成类似结构：

```text
runtime/nnunet/
└── nnUNet_results/
    └── Dataset137_BraTS2021/
        └── nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__3d_fullres/
            ├── dataset.json
            ├── plans.json
            └── fold_0/
                └── checkpoint_final.pth
```

然后同步修改 `.env` 中的 trainer 和 folds，重启 FastAPI。访问
`GET /api/v1/health`，确认 `analysis.ready` 为 `true` 后再从前端执行 AI 分析。

## 结果口径

- 以 `fold_*/validation/summary.json` 的数值为真实验证结果；
- 实习演示可以报告 fold 0 的 WT、TC、ET Dice，但必须注明单折和训练轮数；
- 未完成五折时不能声称完成五折交叉验证；
- 合成 NIfTI 只用于接口和可视化测试，不能用于训练或模型效果证明；
- 该系统是医学影像辅助研究 Demo，不是诊断系统。

## 参考

- nnU-Net 官方使用文档：
  <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md>
- nnU-Net 官方安装文档：
  <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/getting-started/installation-and-setup.md>
- 官方短训练 trainer：
  <https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py>
