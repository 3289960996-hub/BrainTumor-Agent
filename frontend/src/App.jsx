import { useEffect, useMemo, useRef, useState } from "react";

import {
  analyzeCase,
  askAgent,
  checkHealth,
  downloadNifti,
  generateReport,
  uploadMRI,
} from "./api";
import Icon from "./components/Icon";
import MriViewport from "./components/MriViewport";
import UploadDialog from "./components/UploadDialog";
import {
  findTumorSlice,
  parseNiftiBuffer,
  parseNiftiFile,
} from "./nifti";

const MODALITIES = [
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
  ["flair", "FLAIR"],
];

const NAVIGATION = [
  ["analysis", "brain", "影像工作台"],
  ["assistant", "chat", "医学助手"],
];

const QUICK_QUESTIONS = [
  "总结该MRI分析结果",
  "为什么需要关注增强区域？",
  "解释当前肿瘤量化指标",
];

const EMPTY_LAYERS = { wt: true, tc: true, et: true };

function statusText(status) {
  const labels = {
    idle: "等待上传",
    uploading: "上传中",
    uploaded: "等待分析",
    analyzing: "AI分析中",
    analyzed: "分析完成",
  };
  return labels[status] || labels.idle;
}

function metricValue(metrics, key, suffix = "") {
  const value = metrics?.[key];
  if (value === undefined || value === null) {
    return "—";
  }
  return `${value}${suffix}`;
}

function normalizeLocation(location) {
  if (!location) {
    return "—";
  }
  const translations = {
    "left frontal": "左侧额叶",
    "right frontal": "右侧额叶",
    "left temporal": "左侧颞叶",
    "right temporal": "右侧颞叶",
    "left parietal": "左侧顶叶",
    "right parietal": "右侧顶叶",
    "left occipital": "左侧枕叶",
    "right occipital": "右侧枕叶",
  };
  return translations[location] || location;
}

function App() {
  const analysisRef = useRef(null);
  const assistantRef = useRef(null);
  const [caseId, setCaseId] = useState("");
  const [status, setStatus] = useState("idle");
  const [activeNav, setActiveNav] = useState("analysis");
  const [activeTab, setActiveTab] = useState("report");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [volumes, setVolumes] = useState({});
  const [mask, setMask] = useState(null);
  const [maskUrl, setMaskUrl] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [slice, setSlice] = useState(0);
  const [opacity, setOpacity] = useState(0.58);
  const [layers, setLayers] = useState(EMPTY_LAYERS);
  const [report, setReport] = useState("");
  const [messages, setMessages] = useState([]);
  const [citations, setCitations] = useState([]);
  const [question, setQuestion] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [capabilities, setCapabilities] = useState(null);
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    refreshApiStatus();
  }, []);

  const maxSlice = useMemo(() => {
    const depths = Object.values(volumes)
      .filter(Boolean)
      .map((volume) => volume.depth);
    return depths.length ? Math.max(...depths) - 1 : 0;
  }, [volumes]);

  const primaryVolume =
    volumes.t1ce || volumes.flair || volumes.t1 || volumes.t2 || null;
  const isAnalyzed = status === "analyzed";
  const isBusy = Boolean(busyAction) || status === "uploading" || status === "analyzing";
  const unavailableCapabilities = capabilities
    ? [
        !capabilities.analysis && "nnU-Net模型",
        !capabilities.report && "Qwen报告",
        !capabilities.chat && "Qwen问答",
        !capabilities.rag && "RAG知识库",
      ].filter(Boolean)
    : [];

  async function refreshApiStatus() {
    setApiStatus("checking");
    try {
      const health = await checkHealth();
      setCapabilities(health.capabilities || null);
      setApiStatus("connected");
    } catch {
      setCapabilities(null);
      setApiStatus("disconnected");
    }
  }

  function showFeedback(type, message) {
    setFeedback({ type, message });
  }

  async function handleUpload(files, requestedCaseId) {
    setFeedback(null);
    setStatus("uploading");
    setBusyAction("upload");
    try {
      const parsedEntries = await Promise.all(
        MODALITIES.map(async ([key]) => [key, await parseNiftiFile(files[key])]),
      );
      const parsedVolumes = Object.fromEntries(parsedEntries);
      const uploaded = await uploadMRI(files, requestedCaseId);
      setVolumes(parsedVolumes);
      setCaseId(uploaded.case_id);
      setSlice(Math.floor(parsedVolumes.t1.depth / 2));
      setMask(null);
      setMaskUrl("");
      setMetrics(null);
      setReport("");
      setMessages([]);
      setCitations([]);
      setStatus("uploaded");
      setUploadOpen(false);
      setActiveNav("analysis");
      showFeedback("success", "四模态MRI上传完成，可以开始AI分析。");
      return true;
    } catch (uploadError) {
      setStatus("idle");
      showFeedback("error", uploadError.message || "MRI上传失败");
      return false;
    } finally {
      setBusyAction("");
    }
  }

  async function handleAnalyze() {
    if (!caseId) {
      setUploadOpen(true);
      return;
    }
    setFeedback(null);
    setStatus("analyzing");
    setBusyAction("analyze");
    try {
      const result = await analyzeCase(caseId);
      let maskPreviewError = "";
      setMetrics(result.tumor_metrics);
      setMaskUrl(result.mask.download_url);
      try {
        const maskBuffer = await downloadNifti(result.mask.download_url);
        const parsedMask = await parseNiftiBuffer(maskBuffer);
        setMask(parsedMask);
        setSlice(findTumorSlice(parsedMask));
      } catch (maskError) {
        maskPreviewError = maskError.message;
      }
      setStatus("analyzed");
      setActiveNav("analysis");
      if (maskPreviewError) {
        showFeedback(
          "warning",
          `指标分析完成，但mask预览加载失败：${maskPreviewError}`,
        );
      } else {
        showFeedback("success", "AI分割与肿瘤量化分析已完成。");
      }
    } catch (analysisError) {
      setStatus("uploaded");
      showFeedback("error", analysisError.message || "MRI分析失败");
    } finally {
      setBusyAction("");
    }
  }

  async function handleReport() {
    if (!caseId || !isAnalyzed) {
      showFeedback("warning", "请先上传MRI并完成AI分析。");
      navigate("analysis");
      return;
    }
    setActiveTab("report");
    navigate("assistant");
    setFeedback(null);
    setBusyAction("report");
    try {
      const response = await generateReport(caseId);
      setReport(response.report);
      showFeedback("success", "影像辅助报告已生成，请由影像科医师审核。");
    } catch (reportError) {
      showFeedback("error", reportError.message || "辅助报告生成失败");
    } finally {
      setBusyAction("");
    }
  }

  async function handleAsk(explicitQuestion = "") {
    const prompt = (explicitQuestion || question).trim();
    if (!prompt) {
      return;
    }
    setActiveTab("chat");
    navigate("assistant");
    setQuestion("");
    setFeedback(null);
    setMessages((current) => [...current, { role: "user", content: prompt }]);
    setBusyAction("chat");
    try {
      const response = await askAgent(prompt, caseId || null);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer,
          tool: response.tool_name,
        },
      ]);
      setCitations(response.citations || []);
    } catch (chatError) {
      showFeedback("error", chatError.message || "医学助手调用失败");
    } finally {
      setBusyAction("");
    }
  }

  function navigate(target) {
    setActiveNav(target);
    const targetRef = target === "assistant" ? assistantRef : analysisRef;
    requestAnimationFrame(() => {
      targetRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }

  function openAssistantTab(tab) {
    setActiveTab(tab);
    setActiveNav("assistant");
  }

  function toggleLayer(name) {
    setLayers((current) => ({ ...current, [name]: !current[name] }));
  }

  function exportReport() {
    if (!report) {
      return;
    }
    const url = URL.createObjectURL(
      new Blob([report], { type: "text/markdown;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${caseId || "case"}-ai-report.md`;
    anchor.click();
    URL.revokeObjectURL(url);
    showFeedback("success", "辅助报告文本已导出。");
  }

  async function downloadMask() {
    if (!maskUrl || !caseId) {
      showFeedback("warning", "当前病例还没有可下载的分割Mask。");
      return;
    }
    setBusyAction("download");
    setFeedback(null);
    try {
      const buffer = await downloadNifti(maskUrl);
      const url = URL.createObjectURL(
        new Blob([buffer], { type: "application/gzip" }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${caseId}-seg.nii.gz`;
      anchor.click();
      URL.revokeObjectURL(url);
      showFeedback("success", "分割Mask已开始下载。");
    } catch (downloadError) {
      showFeedback("error", downloadError.message || "分割Mask下载失败");
    } finally {
      setBusyAction("");
    }
  }

  return (
    <div className="application">
      <aside className="sidebar">
        <button
          className="brand"
          onClick={() => navigate("analysis")}
          type="button"
        >
          <span className="brand-mark">
            <Icon name="brain" size={22} />
          </span>
          <span>BrainTumor-Agent</span>
        </button>

        <nav aria-label="主导航">
          {NAVIGATION.map(([id, icon, label]) => (
            <button
              className={activeNav === id ? "nav-item active" : "nav-item"}
              key={id}
              onClick={() => navigate(id)}
              type="button"
            >
              <Icon name={icon} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-safety">
          <Icon name="shield" size={18} />
          <span>仅供医学影像辅助，须由专业医师审核</span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <h1>BrainTumor-Agent</h1>
          <div className="case-identity">
            <Icon name="folder" size={17} />
            <span>{caseId ? `Case ${caseId}` : "尚未创建病例"}</span>
          </div>
          <div className={`analysis-status ${status}`}>
            {isAnalyzed && <Icon name="check" size={14} />}
            <span>{statusText(status)}</span>
          </div>
          <div className="top-actions">
            <button
              className={`api-status ${apiStatus}`}
              onClick={refreshApiStatus}
              title={
                unavailableCapabilities.length
                  ? `待配置：${unavailableCapabilities.join("、")}`
                  : "点击重新检查后端连接"
              }
              type="button"
            >
              <span className="status-dot" />
              {apiStatus === "connected"
                ? "后端已连接"
                : apiStatus === "checking"
                  ? "连接检查中"
                  : "后端未连接"}
            </button>
            {maskUrl && (
              <button
                className="compact-action"
                disabled={busyAction === "download"}
                onClick={downloadMask}
                title="下载NIfTI格式分割结果"
                type="button"
              >
                <Icon name="download" size={18} />
                {busyAction === "download" ? "下载中" : "下载 Mask"}
              </button>
            )}
          </div>
        </header>

        <main className="dashboard">
          {feedback && (
            <div
              className={`feedback-banner ${feedback.type}`}
              role={feedback.type === "error" ? "alert" : "status"}
            >
              <span>{feedback.message}</span>
              <button
                aria-label="关闭提示"
                onClick={() => setFeedback(null)}
                type="button"
              >
                <Icon name="close" size={16} />
              </button>
            </div>
          )}
          {!feedback &&
            apiStatus === "connected" &&
            unavailableCapabilities.length > 0 && (
              <div className="readiness-banner" role="status">
                <Icon name="settings" size={16} />
                <span>
                  接口已连接；当前环境待配置：
                  {unavailableCapabilities.join("、")}。上传与页面预览可正常使用。
                </span>
              </div>
            )}

          <section className="analysis-layout" ref={analysisRef}>
            <div className="viewer-area">
              <div className="viewer-grid">
                {MODALITIES.map(([key, title]) => (
                  <MriViewport
                    key={key}
                    layers={layers}
                    mask={mask}
                    opacity={opacity}
                    slice={Math.min(
                      volumes[key]?.depth ? volumes[key].depth - 1 : 0,
                      slice,
                    )}
                    title={title}
                    volume={volumes[key]}
                  />
                ))}
              </div>
              <div className="slice-control">
                <span>切片</span>
                <input
                  disabled={!primaryVolume}
                  max={maxSlice}
                  min="0"
                  onChange={(event) => setSlice(Number(event.target.value))}
                  type="range"
                  value={Math.min(slice, maxSlice)}
                />
                <output>
                  {primaryVolume ? `${slice + 1} / ${maxSlice + 1}` : "—"}
                </output>
              </div>
            </div>

            <aside className="result-panel">
              <div className="panel-title">
                <Icon name="layers" size={18} />
                <h2>AI 定量指标</h2>
              </div>
              <div className="metric-list">
                <Metric
                  color="blue"
                  label="Whole Tumor"
                  value={metricValue(metrics, "tumor_volume", " cm³")}
                />
                <Metric
                  color="coral"
                  label="Tumor Core"
                  value={metricValue(metrics, "tumor_core_volume", " cm³")}
                />
                <Metric
                  color="violet"
                  label="Enhancing Tumor"
                  value={metricValue(metrics, "enhancing_volume", " cm³")}
                />
                <Metric
                  color="teal"
                  label="最大径"
                  value={metricValue(metrics, "max_diameter", " mm")}
                />
                <Metric
                  color="slate"
                  label="位置"
                  value={normalizeLocation(metrics?.location)}
                />
              </div>

              <div className="mask-section">
                <h3>Mask 图层</h3>
                <div className="mask-content">
                  <div className="layer-list">
                    <LayerToggle
                      active={layers.wt}
                      color="wt"
                      label="Whole Tumor"
                      onClick={() => toggleLayer("wt")}
                      short="WT"
                    />
                    <LayerToggle
                      active={layers.tc}
                      color="tc"
                      label="Tumor Core"
                      onClick={() => toggleLayer("tc")}
                      short="TC"
                    />
                    <LayerToggle
                      active={layers.et}
                      color="et"
                      label="Enhancing Tumor"
                      onClick={() => toggleLayer("et")}
                      short="ET"
                    />
                  </div>
                  <div className="mask-preview">
                    <MriViewport
                      layers={layers}
                      mask={mask}
                      opacity={opacity}
                      slice={slice}
                      title=""
                      volume={primaryVolume}
                    />
                  </div>
                </div>
                <label className="opacity-control">
                  <span>透明度</span>
                  <input
                    max="0.85"
                    min="0.1"
                    onChange={(event) => setOpacity(Number(event.target.value))}
                    step="0.01"
                    type="range"
                    value={opacity}
                  />
                  <output>{Math.round(opacity * 100)}%</output>
                </label>
              </div>

              <div className="primary-actions">
                {(status === "idle" || status === "uploading") && (
                  <button
                    className="primary-button"
                    disabled={status === "uploading"}
                    onClick={() => setUploadOpen(true)}
                    type="button"
                  >
                    <Icon name="upload" size={18} />
                    {status === "uploading" ? "正在上传…" : "上传四模态MRI"}
                  </button>
                )}
                {(status === "uploaded" || status === "analyzing") && (
                  <button
                    className="primary-button"
                    disabled={isBusy}
                    onClick={handleAnalyze}
                    type="button"
                  >
                    <Icon name="brain" size={18} />
                    {status === "analyzing" ? "正在分析…" : "开始AI分析"}
                  </button>
                )}
                {isAnalyzed && (
                  <button
                    className="primary-button"
                    disabled={busyAction === "report"}
                    onClick={handleReport}
                    type="button"
                  >
                    <Icon name="report" size={18} />
                    {report ? "重新生成报告" : "生成辅助报告"}
                  </button>
                )}
                {caseId && status !== "uploading" && (
                  <button
                    className="secondary-button"
                    disabled={isBusy}
                    onClick={() => setUploadOpen(true)}
                    type="button"
                  >
                    更换病例
                  </button>
                )}
              </div>
            </aside>
          </section>

          <section className="assistant-panel" ref={assistantRef}>
            <div className="assistant-tabs" role="tablist">
              <TabButton
                active={activeTab === "report"}
                icon="report"
                label="辅助报告"
                onClick={() => openAssistantTab("report")}
              />
              <TabButton
                active={activeTab === "chat"}
                icon="chat"
                label="Agent 问答"
                onClick={() => openAssistantTab("chat")}
              />
              <TabButton
                active={activeTab === "sources"}
                icon="source"
                label="证据来源"
                onClick={() => openAssistantTab("sources")}
              />
            </div>

            {activeTab === "report" && (
              <ReportPanel
                busy={Boolean(busyAction)}
                caseId={caseId}
                isAnalyzed={isAnalyzed}
                metrics={metrics}
                onExport={exportReport}
                onGenerate={handleReport}
                onQuickQuestion={handleAsk}
                report={report}
              />
            )}
            {activeTab === "chat" && (
              <ChatPanel
                busy={busyAction === "chat"}
                messages={messages}
                onAsk={handleAsk}
                onQuestionChange={setQuestion}
                question={question}
              />
            )}
            {activeTab === "sources" && (
              <SourcesPanel citations={citations} />
            )}
          </section>
        </main>
      </div>

      <UploadDialog
        busy={busyAction === "upload"}
        onClose={() => !isBusy && setUploadOpen(false)}
        onSubmit={handleUpload}
        open={uploadOpen}
      />
    </div>
  );
}

function Metric({ color, label, value }) {
  return (
    <div className="metric-row">
      <span className={`metric-icon ${color}`}>
        <Icon name="layers" size={14} />
      </span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LayerToggle({ active, color, label, onClick, short }) {
  return (
    <button
      aria-checked={active}
      className={active ? "layer-toggle active" : "layer-toggle"}
      onClick={onClick}
      role="switch"
      type="button"
    >
      <span className="eye-indicator">{active ? "●" : "○"}</span>
      <span className={`layer-dot ${color}`} />
      <span>
        <strong>{short}</strong>
        <small>{label}</small>
      </span>
    </button>
  );
}

function TabButton({ active, icon, label, onClick }) {
  return (
    <button
      aria-selected={active}
      className={active ? "tab-button active" : "tab-button"}
      onClick={onClick}
      role="tab"
      type="button"
    >
      <Icon name={icon} size={17} />
      {label}
    </button>
  );
}

function ReportPanel({
  busy,
  caseId,
  isAnalyzed,
  metrics,
  onExport,
  onGenerate,
  onQuickQuestion,
  report,
}) {
  return (
    <div className="report-layout">
      <article className="report-preview">
        <div className="content-heading">
          <div>
            <span className="section-kicker">影像辅助报告</span>
            <h3>{caseId ? `Case ${caseId}` : "等待病例数据"}</h3>
          </div>
          <span className="review-badge">需医师审核</span>
        </div>
        {report ? (
          <pre>{report}</pre>
        ) : (
          <div className="report-placeholder">
            <p>
              {isAnalyzed
                ? "AI分割与量化指标已就绪，可生成结构化辅助报告。"
                : "完成MRI上传与AI分析后，可在此生成辅助报告。"}
            </p>
            {metrics && (
              <p>
                当前结果：WT {metrics.tumor_volume} cm³，TC{" "}
                {metrics.tumor_core_volume} cm³，ET{" "}
                {metrics.enhancing_volume} cm³。
              </p>
            )}
          </div>
        )}
      </article>

      <aside className="quick-actions">
        <span className="section-kicker">快捷操作</span>
        <div className="question-chips">
          {QUICK_QUESTIONS.map((item) => (
            <button
              disabled={busy}
              key={item}
              onClick={() => onQuickQuestion(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
        <ul className="pipeline-checks">
          <CheckItem complete={Boolean(caseId)} label="四模态MRI已上传" />
          <CheckItem complete={isAnalyzed} label="nnU-Net分割与量化完成" />
          <CheckItem complete={Boolean(report)} label="辅助报告已生成" />
          <CheckItem complete={Boolean(report)} label="等待影像科医师审核" />
        </ul>
        <div className="report-buttons">
          <button
            className="primary-button"
            disabled={busy}
            onClick={onGenerate}
            type="button"
          >
            {busy
              ? "处理中…"
              : isAnalyzed
                ? "生成报告"
                : "完成分析后生成"}
          </button>
          <button
            className="secondary-button"
            disabled={!report}
            onClick={onExport}
            type="button"
          >
            导出文本
          </button>
        </div>
      </aside>
    </div>
  );
}

function CheckItem({ complete, label }) {
  return (
    <li className={complete ? "complete" : ""}>
      <span>
        {complete && <Icon name="check" size={12} strokeWidth={2.4} />}
      </span>
      {label}
    </li>
  );
}

function ChatPanel({
  busy,
  messages,
  onAsk,
  onQuestionChange,
  question,
}) {
  return (
    <div className="chat-layout">
      <div className="conversation">
        {messages.length ? (
          messages.map((message, index) => (
            <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <span>{message.role === "user" ? "医生" : "MRI Assistant"}</span>
              <p>{message.content}</p>
            </div>
          ))
        ) : (
          <div className="chat-empty">
            <Icon name="chat" size={28} />
            <strong>医学影像辅助助手</strong>
            <p>可以总结当前MRI指标，或查询知识库中的医学影像资料。</p>
          </div>
        )}
      </div>
      <div className="chat-composer">
        <div className="question-chips">
          {QUICK_QUESTIONS.slice(0, 2).map((item) => (
            <button
              disabled={busy}
              key={item}
              onClick={() => onAsk(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            onAsk();
          }}
        >
          <textarea
            maxLength="2000"
            onChange={(event) => onQuestionChange(event.target.value)}
            placeholder="输入关于当前MRI结果或医学影像知识的问题…"
            rows="3"
            value={question}
          />
          <button
            aria-label="发送问题"
            className="send-button"
            disabled={busy || !question.trim()}
            type="submit"
          >
            <Icon name="send" size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}

function SourcesPanel({ citations }) {
  return (
    <div className="sources-panel">
      <div className="content-heading">
        <div>
          <span className="section-kicker">Medical RAG</span>
          <h3>医学资料来源</h3>
        </div>
        <span className="review-badge">需核对原文</span>
      </div>
      {citations.length ? (
        <ol>
          {citations.map((citation, index) => (
            <li key={`${citation}-${index}`}>{citation}</li>
          ))}
        </ol>
      ) : (
        <div className="source-empty">
          <Icon name="source" size={28} />
          <p>在医学问答中完成一次RAG检索后，引用资料将在这里展示。</p>
        </div>
      )}
    </div>
  );
}

export default App;
