import { useEffect, useMemo, useState } from "react";

import {
  analyzeCase,
  applyReportEdit,
  askAgent,
  cancelAnalysisTask,
  cancelComparisonTask,
  checkHealth,
  downloadNifti,
  getCase,
  getComparison,
  getComparisonTask,
  generateReport,
  getAnalysisTask,
  listCases,
  proposeReportEdit,
  startComparisonTask,
  uploadMRI,
} from "./api";
import Icon from "./components/Icon";
import MriViewport from "./components/MriViewport";
import UploadDialog from "./components/UploadDialog";
import {
  findTumorSlice,
  parseNiftiBuffer,
  parseNiftiFile,
  sliceTumorMetrics,
} from "./nifti";

const MODALITIES = [
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
  ["flair", "FLAIR"],
];

const NAVIGATION = [
  ["analysis", "brain", "影像工作台"],
  ["comparison", "compare", "随访对比"],
  ["assistant", "chat", "医学助手"],
];

const QUICK_QUESTIONS = [
  "总结该MRI分析结果",
  "为什么需要关注增强区域？",
  "解释当前肿瘤量化指标",
];

const REPORT_EDIT_ACTIONS = [
  "生成简洁的会诊版报告",
  "只保留病灶位置、体积和水肿信息",
  "优化为规范的影像科表述",
];

const EMPTY_LAYERS = { wt: true, tc: true, et: true };
const ACTIVE_CASE_STORAGE_KEY = "brain-tumor-agent.active-case";
const ACTIVE_COMPARISON_TASK_STORAGE_KEY =
  "brain-tumor-agent.active-comparison-task";
const ACTIVE_ANALYSIS_STATUSES = ["queued", "running", "cancel_requested"];

function isAnalysisTaskActive(task) {
  return Boolean(task?.task_id && ACTIVE_ANALYSIS_STATUSES.includes(task.status));
}

function isComparisonTaskActive(task) {
  return Boolean(task?.task_id && ACTIVE_ANALYSIS_STATUSES.includes(task.status));
}

function navigationFromHash() {
  if (window.location.hash === "#assistant") {
    return "assistant";
  }
  if (window.location.hash === "#comparison") {
    return "comparison";
  }
  return "analysis";
}

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

function sliceMetricValue(metrics, key, suffix, digits) {
  const value = metrics?.[key];
  if (!Number.isFinite(value)) {
    return "—";
  }
  return `${value.toFixed(digits)}${suffix}`;
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
  const [caseId, setCaseId] = useState("");
  const [status, setStatus] = useState("idle");
  const [activeNav, setActiveNav] = useState(navigationFromHash);
  const [activeTab, setActiveTab] = useState("report");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [volumes, setVolumes] = useState({});
  const [mask, setMask] = useState(null);
  const [maskUrl, setMaskUrl] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [metricMode, setMetricMode] = useState("overall");
  const [slice, setSlice] = useState(0);
  const [opacity, setOpacity] = useState(0.58);
  const [layers, setLayers] = useState(EMPTY_LAYERS);
  const [report, setReport] = useState("");
  const [editInstruction, setEditInstruction] = useState("");
  const [editSuggestion, setEditSuggestion] = useState(null);
  const [messages, setMessages] = useState([]);
  const [citations, setCitations] = useState([]);
  const [question, setQuestion] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [capabilities, setCapabilities] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [analysisTask, setAnalysisTask] = useState(null);
  const [comparisonCases, setComparisonCases] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparisonTask, setComparisonTask] = useState(null);
  const [comparisonSpatial, setComparisonSpatial] = useState(null);
  const [comparisonSlice, setComparisonSlice] = useState(0);
  const [comparisonRegion, setComparisonRegion] = useState("wt");
  const [patientGroupId, setPatientGroupId] = useState("");
  const [baselineCaseId, setBaselineCaseId] = useState("");
  const [followupCaseId, setFollowupCaseId] = useState("");
  const [baselineStudyDate, setBaselineStudyDate] = useState("");
  const [followupStudyDate, setFollowupStudyDate] = useState("");

  useEffect(() => {
    refreshApiStatus();
    restoreActiveCase();
    restoreComparisonTask();

    const syncNavigation = () => setActiveNav(navigationFromHash());
    window.addEventListener("hashchange", syncNavigation);
    return () => window.removeEventListener("hashchange", syncNavigation);
  }, []);

  useEffect(() => {
    if (!isAnalysisTaskActive(analysisTask)) {
      return undefined;
    }
    let disposed = false;
    const poll = async () => {
      try {
        const task = await getAnalysisTask(analysisTask.task_id);
        if (disposed) return;
        setAnalysisTask(task);
        if (task.status === "succeeded") {
          await loadCompletedAnalysis(task.case_id);
        } else if (task.status === "failed") {
          setStatus("uploaded");
          showFeedback("error", task.error_message || "MRI分析失败");
        } else if (task.status === "cancelled") {
          setStatus("uploaded");
          showFeedback("warning", "MRI分析任务已取消。");
        }
      } catch (pollError) {
        if (!disposed) {
          showFeedback("error", pollError.message || "无法查询分析任务状态");
        }
      }
    };
    const timer = window.setInterval(poll, 2000);
    poll();
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [analysisTask?.task_id, analysisTask?.status]);

  useEffect(() => {
    if (!isComparisonTaskActive(comparisonTask)) {
      return undefined;
    }
    let disposed = false;
    const poll = async () => {
      try {
        const task = await getComparisonTask(comparisonTask.task_id);
        if (disposed) return;
        setComparisonTask(task);
        if (task.status === "succeeded") {
          window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
          await loadComparisonResult(task.comparison_id);
        } else if (task.status === "failed") {
          window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
          showFeedback("error", task.error_message || "空间对比执行失败");
        } else if (task.status === "cancelled") {
          window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
          showFeedback("warning", "空间对比任务已取消");
        }
      } catch (pollError) {
        if (!disposed) {
          showFeedback("error", pollError.message || "无法查询空间对比任务");
        }
      }
    };
    const timer = window.setInterval(poll, 2000);
    poll();
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [comparisonTask?.task_id, comparisonTask?.status]);

  useEffect(() => {
    if (activeNav !== "comparison") {
      return;
    }
    refreshComparisonCases();
  }, [activeNav]);

  const maxSlice = useMemo(() => {
    const depths = Object.values(volumes)
      .filter(Boolean)
      .map((volume) => volume.depth);
    return depths.length ? Math.max(...depths) - 1 : 0;
  }, [volumes]);

  const primaryVolume =
    volumes.t1ce || volumes.flair || volumes.t1 || volumes.t2 || null;
  const currentSliceMetrics = useMemo(
    () => sliceTumorMetrics(mask, slice),
    [mask, slice],
  );
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

  async function restoreActiveCase() {
    const requestedCaseId = new URLSearchParams(window.location.search).get("case");
    const savedCaseId =
      requestedCaseId || window.localStorage.getItem(ACTIVE_CASE_STORAGE_KEY);
    if (!savedCaseId) {
      return;
    }
    try {
      const restored = await getCase(savedCaseId);
      const modalityEntries = await Promise.all(
        MODALITIES.map(async ([key]) => {
          const url = restored.modalities?.[key];
          if (!url) {
            throw new Error(`病例缺少${key}模态文件`);
          }
          const buffer = await downloadNifti(url);
          return [key, await parseNiftiBuffer(buffer)];
        }),
      );
      const restoredVolumes = Object.fromEntries(modalityEntries);
      setCaseId(restored.case_id);
      window.localStorage.setItem(ACTIVE_CASE_STORAGE_KEY, restored.case_id);
      setVolumes(restoredVolumes);
      setMetrics(restored.tumor_metrics || null);
      setReport(restored.report || "");
      setAnalysisTask(restored.analysis_task || null);
      setEditSuggestion(null);
      setEditInstruction("");
      setSlice(Math.floor(restoredVolumes.t1.depth / 2));

      if (restored.mask?.download_url) {
        setMaskUrl(restored.mask.download_url);
        const maskBuffer = await downloadNifti(restored.mask.download_url);
        const restoredMask = await parseNiftiBuffer(maskBuffer);
        setMask(restoredMask);
        setSlice(findTumorSlice(restoredMask));
      } else {
        setMask(null);
        setMaskUrl("");
      }

      const taskActive = isAnalysisTaskActive(restored.analysis_task);
      setStatus(
        restored.status === "analyzed" || restored.status === "report_ready"
          ? "analyzed"
          : taskActive
            ? "analyzing"
            : "uploaded",
      );
      showFeedback("success", "已恢复上次病例及其分析结果");
    } catch {
      window.localStorage.removeItem(ACTIVE_CASE_STORAGE_KEY);
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
      window.localStorage.setItem(
        ACTIVE_CASE_STORAGE_KEY,
        uploaded.case_id,
      );
      setSlice(Math.floor(parsedVolumes.t1.depth / 2));
      setMask(null);
      setMaskUrl("");
      setMetrics(null);
      setReport("");
      setEditInstruction("");
      setEditSuggestion(null);
      setMessages([]);
      setCitations([]);
      setAnalysisTask(null);
      setStatus("uploaded");
      setUploadOpen(false);
      navigate("analysis");
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
    try {
      const task = await analyzeCase(caseId);
      setAnalysisTask(task);
      showFeedback("success", "分析任务已提交，可以离开页面或稍后恢复。")
    } catch (analysisError) {
      setStatus("uploaded");
      showFeedback("error", analysisError.message || "MRI分析失败");
    }
  }

  async function loadCompletedAnalysis(completedCaseId) {
    const restored = await getCase(completedCaseId);
    setMetrics(restored.tumor_metrics || null);
    if (restored.mask?.download_url) {
      setMaskUrl(restored.mask.download_url);
      try {
        const maskBuffer = await downloadNifti(restored.mask.download_url);
        const parsedMask = await parseNiftiBuffer(maskBuffer);
        setMask(parsedMask);
        setSlice(findTumorSlice(parsedMask));
        showFeedback("success", "AI分割与肿瘤量化分析已完成。");
      } catch (maskError) {
        showFeedback("warning", `指标分析完成，但mask预览加载失败：${maskError.message}`);
      }
    }
    setStatus("analyzed");
  }

  async function handleCancelAnalysis() {
    if (!analysisTask?.task_id) return;
    try {
      const task = await cancelAnalysisTask(analysisTask.task_id);
      setAnalysisTask(task);
      if (task.status === "cancelled") {
        setStatus(metrics ? "analyzed" : "uploaded");
      }
      showFeedback("warning", task.message);
    } catch (cancelError) {
      showFeedback("error", cancelError.message || "无法取消分析任务");
    }
  }

  async function refreshComparisonCases() {
    try {
      const response = await listCases(true);
      setComparisonCases(response.cases || []);
    } catch (caseError) {
      showFeedback("error", caseError.message || "无法读取已分析病例列表");
    }
  }

  async function restoreComparisonTask() {
    const taskId = window.localStorage.getItem(
      ACTIVE_COMPARISON_TASK_STORAGE_KEY,
    );
    if (!taskId) return;
    try {
      const task = await getComparisonTask(taskId);
      setComparisonTask(task);
      if (task.status === "succeeded") {
        window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
        await loadComparisonResult(task.comparison_id);
      } else if (!isComparisonTaskActive(task)) {
        window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
      }
    } catch {
      window.localStorage.removeItem(ACTIVE_COMPARISON_TASK_STORAGE_KEY);
    }
  }

  async function handleCreateComparison(event) {
    event.preventDefault();
    if (
      !patientGroupId.trim() ||
      !baselineCaseId ||
      !followupCaseId ||
      !baselineStudyDate ||
      !followupStudyDate
    ) {
      showFeedback("warning", "请完整选择病例并填写检查日期");
      return;
    }
    setBusyAction("comparison");
    setFeedback(null);
    try {
      const task = await startComparisonTask({
        patient_group_id: patientGroupId.trim(),
        baseline_case_id: baselineCaseId,
        followup_case_id: followupCaseId,
        baseline_study_date: baselineStudyDate,
        followup_study_date: followupStudyDate,
      });
      setComparisonTask(task);
      setComparisonResult(null);
      setComparisonSpatial(null);
      window.localStorage.setItem(
        ACTIVE_COMPARISON_TASK_STORAGE_KEY,
        task.task_id,
      );
      showFeedback("success", "空间对比任务已提交，可以查看实时进度");
    } catch (comparisonError) {
      setComparisonResult(null);
      setComparisonSpatial(null);
      showFeedback("error", comparisonError.message || "随访对比生成失败");
    } finally {
      setBusyAction("");
    }
  }

  async function loadComparisonResult(comparisonId) {
    const result = await getComparison(comparisonId);
    setComparisonResult(result);
    if (!result.spatial_comparison_available) {
      setComparisonSpatial(null);
      showFeedback("success", "随访定量对比已生成；空间对比当前不可用");
      return;
    }
    try {
      const spatial = result.spatial_comparison;
      const [baselineBuffer, followupBuffer, ...changeBuffers] =
        await Promise.all([
          downloadNifti(spatial.baseline_t1ce_url),
          downloadNifti(spatial.artifacts.registered_followup_t1ce),
          ...["wt", "tc", "et"].map((region) =>
            downloadNifti(spatial.artifacts[`${region}_change`]),
          ),
        ]);
      const [baseline, followup, ...changeMasks] = await Promise.all([
        parseNiftiBuffer(baselineBuffer),
        parseNiftiBuffer(followupBuffer),
        ...changeBuffers.map(parseNiftiBuffer),
      ]);
      setComparisonSpatial({
        baseline,
        followup,
        changes: Object.fromEntries(
          ["wt", "tc", "et"].map((region, index) => [
            region,
            changeMasks[index],
          ]),
        ),
      });
      setComparisonSlice(findTumorSlice(changeMasks[0]));
      showFeedback("success", "随访定量与空间变化对比已生成");
    } catch (previewError) {
      setComparisonSpatial(null);
      showFeedback(
        "warning",
        `定量与空间计算已完成，但空间预览加载失败：${previewError.message}`,
      );
    }
  }

  async function handleCancelComparison() {
    if (!comparisonTask?.task_id) return;
    try {
      const task = await cancelComparisonTask(comparisonTask.task_id);
      setComparisonTask(task);
      showFeedback("warning", task.message);
    } catch (cancelError) {
      showFeedback("error", cancelError.message || "无法取消空间对比任务");
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
      setEditSuggestion(null);
      showFeedback("success", "影像辅助报告已生成，请由影像科医师审核。");
    } catch (reportError) {
      showFeedback("error", reportError.message || "辅助报告生成失败");
    } finally {
      setBusyAction("");
    }
  }

  async function handleReportEdit(instruction = "") {
    const prompt = (instruction || editInstruction).trim();
    if (!caseId || !report || !prompt) {
      showFeedback("warning", "请先生成报告，再输入修改指令");
      return;
    }
    setFeedback(null);
    setBusyAction("report-edit");
    try {
      const suggestion = await proposeReportEdit(caseId, prompt);
      setEditSuggestion(suggestion);
      setEditInstruction("");
      showFeedback("success", "已生成报告修改建议，请审核差异后确认");
    } catch (editError) {
      showFeedback("error", editError.message || "报告修改建议生成失败");
    } finally {
      setBusyAction("");
    }
  }

  async function handleApplyReportEdit() {
    if (!caseId || !editSuggestion) {
      return;
    }
    setFeedback(null);
    setBusyAction("report-apply");
    try {
      const response = await applyReportEdit(caseId, editSuggestion.suggestion_id);
      setReport(response.report);
      setEditSuggestion(null);
      showFeedback("success", `报告修改已保存（版本 ${response.revision_id}）`);
    } catch (applyError) {
      showFeedback("error", applyError.message || "报告修改保存失败");
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
    const targetHash =
      target === "assistant"
        ? "#assistant"
        : target === "comparison"
          ? "#comparison"
          : "#workbench";
    if (window.location.hash !== targetHash) {
      window.location.hash = targetHash;
    }
    requestAnimationFrame(() => {
      window.scrollTo({
        behavior: "smooth",
        top: 0,
      });
    });
  }

  function openAssistantTab(tab) {
    setActiveTab(tab);
    navigate("assistant");
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

        <main className={`dashboard ${activeNav}-page`}>
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

          {activeNav === "analysis" && (
          <section className="analysis-layout">
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
                <div className="panel-title-label">
                  <Icon name="layers" size={18} />
                  <h2>AI 定量指标</h2>
                </div>
                <div className="metric-mode" aria-label="定量指标范围">
                  <button
                    aria-pressed={metricMode === "overall"}
                    className={metricMode === "overall" ? "active" : ""}
                    onClick={() => setMetricMode("overall")}
                    type="button"
                  >
                    总体
                  </button>
                  <button
                    aria-pressed={metricMode === "slice"}
                    className={metricMode === "slice" ? "active" : ""}
                    onClick={() => setMetricMode("slice")}
                    type="button"
                  >
                    当前层
                  </button>
                </div>
              </div>
              <div className="metric-list">
                <Metric
                  color="wt"
                  label={metricMode === "slice" ? "Whole Tumor 面积" : "Whole Tumor"}
                  value={
                    metricMode === "slice"
                      ? sliceMetricValue(
                          currentSliceMetrics,
                          "wholeTumorArea",
                          " cm²",
                          2,
                        )
                      : metricValue(metrics, "tumor_volume", " cm³")
                  }
                />
                <Metric
                  color="tc"
                  label={metricMode === "slice" ? "Tumor Core 面积" : "Tumor Core"}
                  value={
                    metricMode === "slice"
                      ? sliceMetricValue(
                          currentSliceMetrics,
                          "tumorCoreArea",
                          " cm²",
                          2,
                        )
                      : metricValue(metrics, "tumor_core_volume", " cm³")
                  }
                />
                <Metric
                  color="et"
                  label={
                    metricMode === "slice"
                      ? "Enhancing Tumor 面积"
                      : "Enhancing Tumor"
                  }
                  value={
                    metricMode === "slice"
                      ? sliceMetricValue(
                          currentSliceMetrics,
                          "enhancingTumorArea",
                          " cm²",
                          2,
                        )
                      : metricValue(metrics, "enhancing_volume", " cm³")
                  }
                />
                <Metric
                  color="teal"
                  label={metricMode === "slice" ? "层内最大径" : "最大径"}
                  value={
                    metricMode === "slice"
                      ? sliceMetricValue(
                          currentSliceMetrics,
                          "maximumDiameter",
                          " mm",
                          1,
                        )
                      : metricValue(metrics, "max_diameter", " mm")
                  }
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
                {status === "analyzing" && analysisTask && (
                  <div className="analysis-progress" role="status">
                    <div>
                      <strong>{analysisTask.message}</strong>
                      <span>{analysisTask.progress}%</span>
                    </div>
                    <progress max="100" value={analysisTask.progress} />
                    <button
                      className="secondary-button"
                      disabled={analysisTask.status === "cancel_requested"}
                      onClick={handleCancelAnalysis}
                      type="button"
                    >
                      {analysisTask.status === "cancel_requested" ? "正在取消" : "取消任务"}
                    </button>
                  </div>
                )}
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
                    disabled={status === "analyzing" || isBusy}
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
          )}

          {activeNav === "assistant" && (
          <section className="assistant-panel">
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
                label="报告协作Agent"
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
                editInstruction={editInstruction}
                editSuggestion={editSuggestion}
                isAnalyzed={isAnalyzed}
                metrics={metrics}
                onApplyEdit={handleApplyReportEdit}
                onEditInstructionChange={setEditInstruction}
                onRejectEdit={() => setEditSuggestion(null)}
                onExport={exportReport}
                onEdit={handleReportEdit}
                onGenerate={handleReport}
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
          )}

          {activeNav === "comparison" && (
            <LongitudinalPanel
              baselineCaseId={baselineCaseId}
              baselineStudyDate={baselineStudyDate}
              busy={
                busyAction === "comparison" ||
                isComparisonTaskActive(comparisonTask)
              }
              cases={comparisonCases}
              followupCaseId={followupCaseId}
              followupStudyDate={followupStudyDate}
              onBaselineCaseChange={setBaselineCaseId}
              onBaselineDateChange={setBaselineStudyDate}
              onCreate={handleCreateComparison}
              onCancel={handleCancelComparison}
              onFollowupCaseChange={setFollowupCaseId}
              onFollowupDateChange={setFollowupStudyDate}
              onPatientGroupChange={setPatientGroupId}
              onRefresh={refreshComparisonCases}
              patientGroupId={patientGroupId}
              result={comparisonResult}
              task={comparisonTask}
              spatial={comparisonSpatial}
              spatialRegion={comparisonRegion}
              spatialSlice={comparisonSlice}
              onSpatialRegionChange={setComparisonRegion}
              onSpatialSliceChange={setComparisonSlice}
            />
          )}
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

function LongitudinalPanel({
  baselineCaseId,
  baselineStudyDate,
  busy,
  cases,
  followupCaseId,
  followupStudyDate,
  onBaselineCaseChange,
  onBaselineDateChange,
  onCancel,
  onCreate,
  onFollowupCaseChange,
  onFollowupDateChange,
  onPatientGroupChange,
  onRefresh,
  patientGroupId,
  result,
  task,
  spatial,
  spatialRegion,
  spatialSlice,
  onSpatialRegionChange,
  onSpatialSliceChange,
}) {
  const ready =
    patientGroupId.trim() &&
    baselineCaseId &&
    followupCaseId &&
    baselineStudyDate &&
    followupStudyDate &&
    baselineCaseId !== followupCaseId;

  return (
    <section className="comparison-layout">
      <aside className="comparison-controls">
        <div className="comparison-heading">
          <span className="section-kicker">Longitudinal MRI</span>
          <h2>随访定量对比</h2>
        </div>
        <form onSubmit={onCreate}>
          <label>
            <span>去标识化检查组</span>
            <input
              maxLength="64"
              onChange={(event) => onPatientGroupChange(event.target.value)}
              pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
              placeholder="例如 subject-001"
              value={patientGroupId}
            />
          </label>
          <fieldset>
            <legend>基线检查</legend>
            <label>
              <span>病例</span>
              <select
                onChange={(event) => onBaselineCaseChange(event.target.value)}
                value={baselineCaseId}
              >
                <option value="">选择已分析病例</option>
                {cases.map((item) => (
                  <option disabled={item.case_id === followupCaseId} key={item.case_id} value={item.case_id}>
                    {item.case_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>检查日期</span>
              <input
                onChange={(event) => onBaselineDateChange(event.target.value)}
                type="date"
                value={baselineStudyDate}
              />
            </label>
          </fieldset>
          <fieldset>
            <legend>随访检查</legend>
            <label>
              <span>病例</span>
              <select
                onChange={(event) => onFollowupCaseChange(event.target.value)}
                value={followupCaseId}
              >
                <option value="">选择已分析病例</option>
                {cases.map((item) => (
                  <option disabled={item.case_id === baselineCaseId} key={item.case_id} value={item.case_id}>
                    {item.case_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>检查日期</span>
              <input
                min={baselineStudyDate || undefined}
                onChange={(event) => onFollowupDateChange(event.target.value)}
                type="date"
                value={followupStudyDate}
              />
            </label>
          </fieldset>
          <div className="comparison-actions">
            <button className="primary-button" disabled={!ready || busy} type="submit">
              <Icon name="compare" size={17} />
              {busy ? `正在计算 ${task?.progress || 0}%` : "生成定量对比"}
            </button>
            <button className="secondary-button" disabled={busy} onClick={onRefresh} type="button">
              刷新病例
            </button>
          </div>
          {isComparisonTaskActive(task) && (
            <div className="comparison-progress" aria-live="polite">
              <div>
                <span>{task.message}</span>
                <strong>{task.progress}%</strong>
              </div>
              <progress max="100" value={task.progress} />
              <button className="secondary-button" onClick={onCancel} type="button">
                取消任务
              </button>
            </div>
          )}
        </form>
        <div className="comparison-scope">
          <Icon name="shield" size={17} />
          <span>仅比较自动分割定量结果，不作进展或疗效判断。</span>
        </div>
      </aside>

      <div className="comparison-results">
        {result ? (
          <>
            <header className="comparison-result-header">
              <div>
                <span className="section-kicker">{result.patient_group_id}</span>
                <h2>{result.interval_days} 天随访变化</h2>
              </div>
              <span className="review-badge">需医师审核</span>
            </header>
            <div className="comparison-summary">
              <div>
                <span>基线</span>
                <strong>{result.baseline_case_id}</strong>
                <small>{result.baseline_study_date}</small>
              </div>
              <Icon name="compare" size={22} />
              <div>
                <span>随访</span>
                <strong>{result.followup_case_id}</strong>
                <small>{result.followup_study_date}</small>
              </div>
            </div>
            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>指标</th>
                    <th>基线</th>
                    <th>随访</th>
                    <th>变化量</th>
                    <th>变化率</th>
                    <th>方向</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metrics.map((metric) => (
                    <tr key={metric.key}>
                      <th>{metric.label}</th>
                      <td>{formatComparisonValue(metric.baseline, metric.unit)}</td>
                      <td>{formatComparisonValue(metric.followup, metric.unit)}</td>
                      <td className={`change ${metric.direction}`}>
                        {formatSignedChange(metric)}
                      </td>
                      <td>
                        {metric.percent_change === null
                          ? "—"
                          : `${signed(metric.percent_change)}%`}
                      </td>
                      <td>{directionLabel(metric.direction)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="location-comparison">
              <span>主要位置</span>
              <strong>{normalizeLocation(result.baseline_location)}</strong>
              <Icon name="compare" size={16} />
              <strong>{normalizeLocation(result.followup_location)}</strong>
              <small>{result.location_consistent ? "位置描述一致" : "位置描述存在差异"}</small>
            </div>
            {result.spatial_comparison_available && spatial ? (
              <SpatialComparisonViewer
                data={result.spatial_comparison}
                onRegionChange={onSpatialRegionChange}
                onSliceChange={onSpatialSliceChange}
                region={spatialRegion}
                slice={spatialSlice}
                volumes={spatial}
              />
            ) : (
              <div className="comparison-limit">
                空间对比不可用：
                {result.spatial_comparison?.unavailable_reason ||
                  result.spatial_comparison?.quality?.warnings?.join("；") ||
                  "缺少配准影像或配准质控未通过"}
              </div>
            )}
          </>
        ) : (
          <div className="comparison-empty">
            <Icon name="compare" size={32} />
            <strong>尚未生成随访对比</strong>
            <p>可用已完成定量分析的病例：{cases.length} 例</p>
          </div>
        )}
      </div>
    </section>
  );
}

function SpatialComparisonViewer({
  data,
  onRegionChange,
  onSliceChange,
  region,
  slice,
  volumes,
}) {
  const regionChanges = data.changes[region];
  const maxSlice = Math.max(0, volumes.baseline.depth - 1);
  return (
    <section className="spatial-comparison">
      <header>
        <div>
          <span className="section-kicker">Rigid registration QC passed</span>
          <h3>空间变化对比</h3>
        </div>
        <div className="spatial-region-tabs" aria-label="肿瘤区域">
          {["wt", "tc", "et"].map((item) => (
            <button
              className={region === item ? "active" : ""}
              key={item}
              onClick={() => onRegionChange(item)}
              type="button"
            >
              {item.toUpperCase()}
            </button>
          ))}
        </div>
      </header>
      <div className="spatial-viewports">
        <MriViewport
          changeMask={volumes.changes[region]}
          layers={EMPTY_LAYERS}
          opacity={0.62}
          slice={slice}
          title="基线 T1ce"
          volume={volumes.baseline}
        />
        <MriViewport
          changeMask={volumes.changes[region]}
          layers={EMPTY_LAYERS}
          opacity={0.62}
          slice={slice}
          title="配准后随访 T1ce"
          volume={volumes.followup}
        />
      </div>
      <div className="spatial-slice-control">
        <span>切片 {slice + 1} / {maxSlice + 1}</span>
        <input
          max={maxSlice}
          min="0"
          onChange={(event) => onSliceChange(Number(event.target.value))}
          type="range"
          value={Math.min(slice, maxSlice)}
        />
      </div>
      <div className="change-legend">
        <span><i className="resolved" />消退 {regionChanges.resolved_volume_cm3.toFixed(2)} cm³</span>
        <span><i className="persistent" />持续 {regionChanges.persistent_volume_cm3.toFixed(2)} cm³</span>
        <span><i className="new" />新增 {regionChanges.new_volume_cm3.toFixed(2)} cm³</span>
      </div>
      <div className="registration-quality">
        <span>相关性 {data.quality.correlation_before.toFixed(3)} → {data.quality.correlation_after.toFixed(3)}</span>
        <span>脑区重叠 {data.quality.foreground_dice.toFixed(3)}</span>
        <span>仅刚性配准</span>
      </div>
    </section>
  );
}

function formatComparisonValue(value, unit) {
  const suffix = unit === "cm3" ? " cm³" : unit === "%" ? "%" : ` ${unit}`;
  return `${Number(value).toFixed(unit === "mm" ? 1 : 2)}${suffix}`;
}

function signed(value) {
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatSignedChange(metric) {
  if (metric.percentage_point_change !== null) {
    return `${signed(metric.percentage_point_change)} 个百分点`;
  }
  const unit = metric.unit === "cm3" ? "cm³" : metric.unit;
  return `${signed(metric.absolute_change)} ${unit}`;
}

function directionLabel(direction) {
  return {
    increased: "增加",
    decreased: "减少",
    unchanged: "不变",
  }[direction];
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
  editInstruction,
  editSuggestion,
  isAnalyzed,
  metrics,
  onApplyEdit,
  onEditInstructionChange,
  onRejectEdit,
  onExport,
  onEdit,
  onGenerate,
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
          {REPORT_EDIT_ACTIONS.map((item) => (
            <button
              disabled={busy}
              key={item}
              onClick={() => onEdit(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </div>
        <form
          className="report-edit-composer"
          onSubmit={(event) => {
            event.preventDefault();
            onEdit();
          }}
        >
          <label htmlFor="report-edit-instruction">医生修改指令</label>
          <textarea
            disabled={!report || busy}
            id="report-edit-instruction"
            maxLength="2000"
            onChange={(event) => onEditInstructionChange(event.target.value)}
            placeholder="例如：改成简洁会诊版，保留定量指标和人工复核提示"
            rows="3"
            value={editInstruction}
          />
          <button
            className="secondary-button"
            disabled={!report || busy || !editInstruction.trim()}
            type="submit"
          >
            {busy ? "正在生成建议…" : "生成修改建议"}
          </button>
        </form>
        {editSuggestion && (
          <div className="report-edit-suggestion" role="status">
            <div className="edit-suggestion-heading">
              <strong>待医生确认的修改建议</strong>
              <span>核心定量指标由后端锁定</span>
            </div>
            <ul>
              {editSuggestion.change_summary.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <div className="edit-diff">
              <div>
                <span>当前报告</span>
                <pre>{editSuggestion.current_report}</pre>
              </div>
              <div>
                <span>修改建议</span>
                <pre>{editSuggestion.proposed_report}</pre>
              </div>
            </div>
            <div className="report-edit-buttons">
              <button
                className="primary-button"
                disabled={busy}
                onClick={onApplyEdit}
                type="button"
              >
                确认并保存新版本
              </button>
              <button
                className="secondary-button"
                disabled={busy}
                onClick={onRejectEdit}
                type="button"
              >
                保留当前报告
              </button>
            </div>
          </div>
        )}
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
