# MRI 分割结果分析

`analyzer.py` 接收 nnU-Net 输出的三维 NIfTI mask，生成便于 RAG 和 LLM Agent
直接消费的标准 JSON。

## 指标定义

- `tumor_volume`：Whole Tumor 体积，单位为 cm³。
- `location`：主要肿瘤位置。默认在 RAS+ 标准方向中进行左右半球和脑叶粗定位。
- `enhancing_ratio`：Enhancing Tumor 体素数除以 Whole Tumor 体素数。
- `edema`：是否存在水肿区域。

默认接受推理模块恢复后的 BraTS 标签：

- `0`：背景
- `1`：坏死/非增强核心
- `2`：水肿
- `4`：增强肿瘤

如果直接分析 nnU-Net 内部的 `0/1/2/3` 输出，请使用
`--label-space nnunet`。

## 运行

在项目根目录执行：

```powershell
python -m feature_extract.analyzer `
  --mask "D:\BraTS_predictions\brats_predictions\BraTS2021_00000.nii.gz" `
  --output "D:\BraTS_predictions\features\BraTS2021_00000.json"
```

标准输出：

```json
{
  "tumor_volume": 35.5,
  "location": "left frontal",
  "enhancing_ratio": 0.42,
  "edema": true
}
```

Python 调用：

```python
from feature_extract.analyzer import analyze_mask

result = analyze_mask("BraTS2021_00000.nii.gz", label_space="brats")
agent_payload = result.to_dict()
```

## 使用脑区 atlas

默认位置结果是粗粒度近似，不能替代标准解剖配准。若已有与 mask 完全同空间的
标签 atlas，可提供 atlas 和标签映射，分析器将选择与 Whole Tumor 重叠最多的区域：

```powershell
python -m feature_extract.analyzer `
  --mask "BraTS2021_00000.nii.gz" `
  --atlas "lobar_atlas.nii.gz" `
  --atlas-label-map "lobar_labels.json"
```

`lobar_labels.json` 示例：

```json
{
  "1": "left frontal",
  "2": "right frontal",
  "3": "left temporal",
  "4": "right temporal"
}
```

atlas 必须与 mask 的 shape 和 affine 一致；离散 atlas 的配准应使用最近邻插值。

## 完整肿瘤量化

`tumor_measure.py`在基础分析之上增加 Tumor Core、Enhancing Tumor、水肿体积、
区域占比和 Whole Tumor 三维最大径：

```powershell
python -m feature_extract.tumor_measure `
  --mask "D:\BraTS_predictions\brats_predictions\BraTS2021_00000.nii.gz" `
  --output "D:\BraTS_predictions\features\BraTS2021_00000_measurement.json"
```

默认 JSON 同时包含基础字段和相对 Whole Tumor 的量化比例：

```json
{
  "tumor_volume": 35.5,
  "tumor_core_volume": 12.1,
  "enhancing_volume": 4.8,
  "max_diameter": 46.2,
  "edema": true,
  "location": "left frontal",
  "edema_volume": 23.4,
  "tumor_core_ratio": 0.3408,
  "enhancing_ratio": 0.1352,
  "edema_ratio": 0.6592
}
```

其中：

- `tumor_volume`等于 Whole Tumor 体积，单位为 cm³。
- `max_diameter`是 Whole Tumor 边界体素中心在三维物理空间中的最大距离，单位为 mm。
- 三个 ratio 均以 Whole Tumor 体素数为分母。
- 仅凭 segmentation mask 无法可靠计算肿瘤占完整脑实质的比例；该指标需要额外 brain mask。

如需严格保持六字段 JSON，添加`--summary-only`。Python/Agent 可直接调用：

```python
from feature_extract.tumor_measure import measure_tumor

result = measure_tumor("BraTS2021_00000.nii.gz")
agent_payload = result.to_dict()
```
