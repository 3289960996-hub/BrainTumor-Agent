# BrainTumor MRI Assistant Agent

本模块使用LangGraph编排三个白名单工具，并通过Qwen-plus完成受控的意图分类、
影像摘要和基于证据的医学知识回答。它的定位是医学影像辅助助手，不是诊断模型。

## 工作流

```text
医生问题
  │
  ├─ 总结/解释当前分析结果 ─ MRI Analyzer ─ Qwen影像摘要 ─ 安全校验
  ├─ 生成影像辅助报告 ───── Report Generator ────────── 安全校验
  └─ 指南/影像学意义问题 ── Medical RAG ─ Qwen证据回答 ─ 安全校验
```

高置信关键词由本地规则路由，其他问题才交给Qwen进行白名单意图分类。模型不能自由
选择或构造工具，只能进入预定义的LangGraph分支。

## 三个工具

- `MRIAnalyzerTool`：校验feature JSON，解释位置、Whole Tumor体积、增强区域占比、
  水肿及可选定量指标。它不推断病理类型或WHO分级。
- `ReportGeneratorTool`：复用`report/generator.py`生成五章节MRI影像辅助报告。
- `MedicalRAGTool`：复用`rag/retriever.py`，返回带资料版本、页码和相关度的证据。

总结和报告请求必须提供至少包含以下字段的JSON：

```json
{
  "location": "left frontal",
  "tumor_volume": 35.5,
  "enhancing_ratio": 0.228,
  "edema": true
}
```

还可包含`tumor_core_volume`、`enhancing_volume`、`max_diameter`、
`edema_volume`、`tumor_core_ratio`和`edema_ratio`。

## 配置

复制`.env.example`并设置：

```powershell
$env:DASHSCOPE_API_KEY = "your-api-key"
$env:BTA_QWEN_MODEL = "qwen-plus"
$env:BTA_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:BTA_QWEN_ENABLE_DATA_INSPECTION = "false"
$env:BTA_FAISS_INDEX_PATH = "runtime\knowledge\faiss"
```

生产环境应由Secret Manager注入API Key，不要把密钥或可识别患者信息写入代码、
日志、提示词模板或版本库。

## 运行

总结当前MRI分析结果：

```powershell
python -m agent.assistant `
  --question "总结该MRI分析结果" `
  --feature-json "runtime\cases\case-001\features.json" `
  --case-id "case-001"
```

查询增强区域的影像学意义：

```powershell
python -m agent.assistant `
  --question "为什么需要关注增强区域？" `
  --rag-index "runtime\knowledge\faiss"
```

在命令末尾增加`--json`可返回适合FastAPI和React消费的结构化响应，包括
`intent`、`tool_name`、`citations`、`safety_warnings`和
`requires_human_review`。

## 安全边界

- 仅总结输入的AI分割和定量指标，不补充输入中不存在的影像表现。
- RAG回答只能依据检索证据，关键表述应保留`[资料1]`等编号及来源。
- 后处理会拦截直接确诊、病理类型推断和WHO分级式表达；修复失败时停止原回答。
- 所有输出均标记需要专业医师结合原始影像、临床资料和必要检查审核。
- 知识库为空、Qwen调用失败或feature JSON无效时返回受控错误，不自动给出通用诊断。

## 测试

测试不访问Qwen API，也不加载BGE模型：

```powershell
pytest agent/tests -q
```
