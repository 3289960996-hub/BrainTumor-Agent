const DEFAULT_API_BASE = "/api/v1";

const API_BASE = (
  import.meta.env?.VITE_API_BASE_URL || DEFAULT_API_BASE
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`, {
    headers: { Accept: "application/json" },
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      payload?.detail?.message ||
      payload?.detail ||
      payload?.message ||
      `请求失败（HTTP ${response.status}）`;
    throw new ApiError(String(message), response.status);
  }
  return payload;
}

export async function uploadMRI(files, requestedCaseId = "") {
  const form = new FormData();
  Object.entries(files).forEach(([modality, file]) => {
    form.append(modality, file);
  });
  if (requestedCaseId.trim()) {
    form.append("case_id", requestedCaseId.trim());
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });
  return parseResponse(response);
}

export async function analyzeCase(caseId) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId }),
  });
  return parseResponse(response);
}

export async function getCase(caseId) {
  const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}`, {
    headers: { Accept: "application/json" },
  });
  return parseResponse(response);
}

export async function generateReport(caseId) {
  const response = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId }),
  });
  return parseResponse(response);
}

export async function proposeReportEdit(caseId, instruction) {
  const response = await fetch(`${API_BASE}/report/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId, instruction: instruction.trim() }),
  });
  return parseResponse(response);
}

export async function applyReportEdit(caseId, suggestionId) {
  const response = await fetch(`${API_BASE}/report/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId, suggestion_id: suggestionId }),
  });
  return parseResponse(response);
}

export async function askAgent(question, caseId = null) {
  const payload = { question: question.trim() };
  if (caseId) {
    payload.case_id = caseId;
  }
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function downloadNifti(url) {
  const artifactUrl = new URL(url, window.location.origin);
  const sameOriginPath = `${artifactUrl.pathname}${artifactUrl.search}`;
  const response = await fetch(sameOriginPath);
  if (!response.ok) {
    throw new ApiError("无法下载分割mask", response.status);
  }
  return response.arrayBuffer();
}
