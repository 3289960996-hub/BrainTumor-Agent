# UPENN-GBM十病例五折GPU验证

本流程验证已安装的`Dataset002_BRATS19`五折权重，不用于训练模型。为避免旧
BraTS/MSD来源可能与训练集重叠，验证队列选自独立机构UPENN-GBM。

## 数据与许可

- 数据集：UPENN-GBM，The Cancer Imaging Archive（TCIA）。
- DOI：`10.7937/TCIA.709X-DN49`。
- 数据许可：CC BY 4.0。
- 真值：`images_segm`专家修正分割标签。
- 队列：10个基线病例，四模态MRI与标签的shape、spacing和affine均已核验。
- 标签：BraTS标准`0/1/2/4`。

仓库不包含MRI原始数据、模型权重或预测Mask。

## 准备输入

使用以下脚本准备和检查验证病例，并生成病例输入包：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_upenn_validation.py `
  --colab-zip runtime\colab\BrainTumor-Agent-UPENN-10-independent-validation-cases.zip
```

提交当前源码后生成Colab源码包，再重新生成或核对Notebook：

```powershell
git archive --format=zip --prefix=BrainTumor-Agent/ `
  --output=runtime\colab\BrainTumor-Agent-source-multicase.zip HEAD
.\.venv\Scripts\python.exe scripts\build_upenn_validation_notebook.py
```

将以下文件放入Google Drive的
`MyDrive/BrainTumor-Agent-Colab/inputs/`：

- `BrainTumor-Agent-source-multicase.zip`
- `Dataset002_BRATS19.zip`
- `BrainTumor-Agent-UPENN-10-independent-validation-cases.zip`

这些压缩包均为本地或Drive运行输入，不提交到Git。

## Colab运行

1. 打开
   `notebooks/BrainTumor_Agent_UPENN_10_GPU_Validation_Colab.ipynb`。
2. 将运行时类型设为T4 GPU。
3. 挂载Google Drive并点击“全部运行”。
4. Notebook使用fold 0、1、2、3、4集成推理。
5. 模型输出按`brats19_preserved`恢复为BraTS标准标签。

成功后Drive的`results/`目录包含：

- `BrainTumor-Agent-UPENN-10-five-fold-GPU-validation.zip`
- `evaluation-upenn-10-cases.json`
- `evaluation-upenn-10-cases.csv`

## 本地验收

下载结果后运行：

```powershell
.\.venv\Scripts\python.exe scripts\archive_upenn_validation_results.py `
  --results-dir "E:\下载结果目录"
```

脚本检查10例队列、逐病例Dice、宏平均、五折与CUDA元数据、预测Mask、ZIP完整性和
JSON/CSV一致性。完整大文件归档只保存在本地`artifacts/`中。

## 已完成结果

2026-08-07在Tesla T4完成验证：

- nnU-Net：`2.8.1`
- folds：`0,1,2,3,4`
- 输出标签配置：`brats19_preserved`
- WT Dice：`0.895287`
- TC Dice：`0.918673`
- ET Dice：`0.862139`

逐病例指标和可复核元数据见
[`validation/upenn-10/`](validation/upenn-10/README.md)。该结果仅为10例独立机构
小样本验证，不能作为临床级验证或部署依据。
