# FastAPI MRI Assistant后端

后端通过服务层复用现有MRI处理、nnU-Net、定量分析、报告和Agent模块。默认API前缀为
`/api/v1`，因此四个核心接口分别是：

- `POST /api/v1/upload`
- `POST /api/v1/analyze`
- `POST /api/v1/report`
- `POST /api/v1/chat`

此外提供`GET /api/v1/cases/{case_id}/mask`下载BraTS标签空间的NIfTI mask。

## 处理流程

```text
/upload
  └─ 四模态流式落盘 + NIfTI头检查

/analyze
  └─ 尺寸/spacing检查 + MONAI归一化
     └─ nnU-Net V2 GPU推理
        └─ BraTS标签还原
           └─ WT/TC/ET/水肿、体积、最大径和位置计算

/report
  └─ 读取features.json → Qwen-plus辅助报告 → 本地安全校验

/chat
  └─ 注入当前病例指标 → LangGraph Agent → MRI/RAG/报告工具
```

病例产物保存在`BTA_DATA_ROOT/cases/{case_id}`。服务端默认生成去标识化case ID；
如果调用方自行提供case ID，只允许字母、数字、下划线和连字符，禁止使用姓名、住院号
等直接身份标识。

## 配置

复制`.env.example`并至少配置：

```dotenv
BTA_DATA_ROOT=./runtime/data
BTA_MAX_UPLOAD_SIZE_MB=1024
NNUNET_ROOT=./runtime/nnunet
BTA_NNUNET_DEVICE=cuda
BTA_NNUNET_GPU_ID=0
DASHSCOPE_API_KEY=
BTA_FAISS_INDEX_PATH=./runtime/knowledge/faiss
```

`NNUNET_ROOT/nnUNet_results`中必须已有与dataset、trainer、plans、fold和checkpoint配置
匹配的训练结果。RAG问答前必须先构建FAISS索引。

## 启动

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app `
  --host 0.0.0.0 `
  --port 8000 `
  --reload
```

交互文档：`http://localhost:8000/docs`

## 上传

上传字段固定为`t1`、`t1ce`、`t2`和`flair`：

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/upload" `
  -F "case_id=case-001" `
  -F "t1=@D:\BraTS\case-001_t1.nii.gz" `
  -F "t1ce=@D:\BraTS\case-001_t1ce.nii.gz" `
  -F "t2=@D:\BraTS\case-001_t2.nii.gz" `
  -F "flair=@D:\BraTS\case-001_flair.nii.gz"
```

## 分析

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/analyze" `
  -H "Content-Type: application/json" `
  -d "{\"case_id\":\"case-001\"}"
```

响应包含mask下载地址和完整肿瘤量化指标。当前接口等待GPU分析完成后返回；单次推理
时间较长。生产部署建议将`MRIAnalysisPipeline`放入Celery/GPU任务队列，并用数据库
锁或分布式锁保证同一病例不会并发推理。

## 报告

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/report" `
  -H "Content-Type: application/json" `
  -d "{\"case_id\":\"case-001\"}"
```

必须先成功调用`/analyze`。报告属于AI影像辅助内容，响应始终包含
`requires_human_review: true`。

## Agent问答

病例总结：

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/chat" `
  -H "Content-Type: application/json" `
  -d "{\"case_id\":\"case-001\",\"question\":\"总结该MRI分析结果\"}"
```

医学知识问题可以不传`case_id`：

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/chat" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"为什么需要关注增强区域？\"}"
```

## 测试

测试通过FastAPI依赖覆盖替换GPU、Qwen和RAG服务，不调用外部API：

```powershell
pytest backend/tests -q
```
