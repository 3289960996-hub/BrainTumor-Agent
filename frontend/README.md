# BrainTumor-Agent前端

React + Vite医生端演示界面，对接FastAPI默认接口前缀`/api/v1`。

## 已实现功能

- T1、T1ce、T2、FLAIR四模态NIfTI上传；支持一次选择4个文件后按文件名自动匹配，
  并要求医生确认对应关系。
- 浏览器直接解析`.nii`和`.nii.gz`，无需把影像转换为PNG。
- 四窗口轴位切片显示及共享切片滑块。
- 加载nnU-Net输出mask，切换WT、TC、ET图层和透明度。
- 显示Whole Tumor、Tumor Core、Enhancing Tumor、最大径和位置。
- 调用Qwen-plus辅助报告接口并导出Markdown。
- 调用MRI Assistant Agent，展示回答和RAG资料来源。

当前浏览器端NIfTI查看器面向BraTS演示数据，按NIfTI原始轴顺序显示轴位切片，不替代
专业DICOM/PACS阅片软件。

## 启动

先启动FastAPI：

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app `
  --reload `
  --host 127.0.0.1 `
  --port 8000
```

再启动前端：

```powershell
cd frontend
npm install
npm run dev
```

访问`http://localhost:5173`。开发服务器会将`/api`代理到
`http://localhost:8000`。

### API密钥与跨域部署

浏览器端**不持有**`BTA_API_KEY`。开发/预览服务器在服务端注入请求头：

```dotenv
# frontend/.env.local（不带VITE_前缀，不会被内联进浏览器代码）
BTA_API_KEY=与后端.env中相同的值
```

`vite.config.js`读取该变量，并由`/api`代理加上`X-API-Key`。**不要**使用
`VITE_API_KEY`等`VITE_`前缀变量：它们会被打包进浏览器产物，任何打开页面的人
都能读出密钥。

前后端部署在不同域名时，同样不要让浏览器持有密钥，应由反向代理注入
`X-API-Key`：

```dotenv
VITE_API_BASE_URL=https://your-api.example.com/api/v1
```

## 操作顺序

1. 点击“上传MRI”，一次选择四个BraTS文件自动匹配，或分别手动选择；核对后勾选确认。
2. 上传完成后点击“开始AI分析”。
3. 查看四模态切片、mask图层及定量指标。
4. 在下方生成辅助报告，或向医学影像助手提问。

辅助报告和Agent回答均不构成疾病确诊，须由专业医师审核。
