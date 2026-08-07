# MRI影像辅助报告生成

该模块读取MRI分析JSON，调用Qwen-plus生成审慎的影像表现总结和建议关注指标，
再由本地模板组装为五章节Markdown报告。

## 医学安全设计

- 检查信息、AI分割结果和数值指标由本地模板确定性生成，模型不能改写。
- Qwen-plus仅生成影像表现总结和建议关注指标。
- 提示词禁止疾病确诊、病理类型、WHO分级、良恶性判断和治疗方案。
- 输出必须使用“影像表现提示”和“建议结合临床”。
- 模型输出经过JSON解析和禁用诊断措辞校验；失败时修复一次，仍不合格则拒绝输出。
- 所有报告都标记为AI辅助结果并要求影像科医师审核。

## 环境变量

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."
$env:BTA_QWEN_MODEL = "qwen-plus"
$env:BTA_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:BTA_QWEN_ENABLE_DATA_INSPECTION = "false"
```

生产环境建议将`BTA_QWEN_BASE_URL`替换为阿里云百炼控制台提供的业务空间专属地址：

```text
https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

API Key与服务地域必须匹配，不要把真实密钥写入代码或提交到Git。

## 输入

基础JSON：

```json
{
  "location": "left frontal",
  "tumor_volume": 35.5,
  "enhancing_ratio": 0.42,
  "edema": true
}
```

也兼容`feature_extract/tumor_measure.py`生成的增强字段，如
`tumor_core_volume`、`enhancing_volume`、`max_diameter`和`edema_volume`。

## 运行

```powershell
python -m report.generator `
  --input-json "D:\BraTS_predictions\features\case_measurement.json" `
  --output "D:\BraTS_predictions\reports\case_report.md" `
  --case-id "BraTS2021_00001"
```

## Agent调用

```python
from report.generator import QwenReportGenerator

generator = QwenReportGenerator()
result = generator.generate(analysis_json, case_id="BraTS2021_00001")
agent_state = result.to_agent_payload()
```

本模块只提供辅助影像文字，不能作为独立诊断或临床决策依据。
