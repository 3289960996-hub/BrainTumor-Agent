# 医学影像知识库RAG

本模块将本地、合法持有的医学PDF转换为带页码引用的FAISS知识索引：

```text
PDF → 按页解析 → 文本清洗 → 递归切分 → BGE-M3 Embedding → FAISS → Retriever
```

知识主题固定为：

- `who_cns`：WHO CNS Tumor Classification
- `nccn_glioma`：NCCN Glioma/CNS Guideline
- `glioma_mri`：MRI胶质瘤影像表现
- `follow_up_criteria`：肿瘤随访评价标准

## 版权与版本

该模块不会自动下载或内置WHO、NCCN等受版权或访问条款约束的资料。请只放入机构
或用户合法获得并被授权用于该场景的PDF。每份PDF必须在manifest中登记真实标题、
版本、来源和主题；指南更新后应重新构建新索引，不要混用不同版本。

推荐目录：

```text
runtime/knowledge/
├── pdfs/
│   ├── who_cns_classification.pdf
│   ├── nccn_glioma_guideline.pdf
│   ├── glioma_mri_review.pdf
│   └── tumor_follow_up_criteria.pdf
├── manifest.json
└── faiss/
```

复制并编辑示例manifest：

```powershell
Copy-Item rag\manifest.example.json runtime\knowledge\manifest.json
```

扫描版PDF没有可提取文本时需要先OCR；当前loader不会静默调用OCR。

## 安装

```powershell
pip install -r requirements.txt
```

当前CPU Demo使用`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，
比BGE-M3更适合无GPU电脑，并能支持中文提问检索英文指南。首次运行会从Hugging
Face下载模型；离线环境可预下载到受控缓存目录并设置：

```powershell
$env:BTA_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
$env:BTA_EMBEDDING_DEVICE = "cpu"
$env:BTA_EMBEDDING_CACHE_DIR = "D:\models\huggingface"
$env:BTA_EMBEDDING_LOCAL_FILES_ONLY = "true"
```

## 构建FAISS索引

```powershell
python -m rag.vector_store `
  --pdf-dir "runtime\knowledge\pdfs" `
  --manifest "runtime\knowledge\manifest.json" `
  --index-dir "runtime\faiss" `
  --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" `
  --device cpu `
  --batch-size 16
```

重新构建同一路径时增加`--overwrite`。索引目录包含：

- `index.faiss`：向量索引
- `index.pkl`：LangChain文档元数据
- `index_manifest.json`：模型、来源文档、chunk数量和文件SHA-256

加载时会先验证索引文件哈希。由于LangChain FAISS的docstore使用pickle，只应加载
本系统从可信PDF构建的索引，禁止接收不可信第三方索引目录。

## 检索示例

```powershell
python -m rag.retriever `
  --index-dir "runtime\faiss" `
  --query "增强区域在胶质瘤评估中的意义？" `
  --top-k 5
```

限定主题：

```powershell
python -m rag.retriever `
  --index-dir "runtime\faiss" `
  --query "随访时如何评价病灶变化？" `
  --topic glioma_mri `
  --json
```

## Agent调用

```python
from rag.retriever import MedicalKnowledgeRetriever

retriever = MedicalKnowledgeRetriever("runtime/faiss")
result = retriever.retrieve("增强区域在胶质瘤评估中的意义？", top_k=5)

for chunk in result.chunks:
    print(chunk.citation, chunk.score)

agent_context = result.format_context()
```

Retriever只返回相关医学资料和出处，不自行生成诊断结论。医生和上层Agent必须核对
原文版本、页码及适用范围。
