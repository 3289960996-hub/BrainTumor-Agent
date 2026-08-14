import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeCase,
  askAgent,
  checkHealth,
  downloadNifti,
  generateReport,
  getAnalysisTask,
  cancelAnalysisTask,
} from "./api.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("前端API方法调用FastAPI对应路由", async (context) => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return jsonResponse({ status: "ok" });
  };

  await checkHealth();
  await analyzeCase("case-001");
  await getAnalysisTask("task-001");
  await cancelAnalysisTask("task-001");
  await generateReport("case-001");
  await askAgent("总结该MRI分析结果", "case-001");

  assert.deepEqual(
    calls.map(({ url }) => url),
    [
      "/api/v1/health",
      "/api/v1/analyze",
      "/api/v1/analysis-tasks/task-001",
      "/api/v1/analysis-tasks/task-001/cancel",
      "/api/v1/report",
      "/api/v1/chat",
    ],
  );
  assert.deepEqual(JSON.parse(calls[1].options.body), {
    case_id: "case-001",
  });
  assert.equal(calls[3].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[4].options.body), {
    case_id: "case-001",
  });
  assert.deepEqual(JSON.parse(calls[5].options.body), {
    question: "总结该MRI分析结果",
    case_id: "case-001",
  });
});

test("Mask下载通过前端同源代理，避免跨域失败", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  let requestedUrl = "";
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
  });

  globalThis.window = { location: { origin: "http://127.0.0.1:5173" } };
  globalThis.fetch = async (url) => {
    requestedUrl = url;
    return new Response(new Uint8Array([1, 2, 3]), { status: 200 });
  };

  const result = await downloadNifti(
    "http://127.0.0.1:8000/api/v1/cases/case-001/mask",
  );

  assert.equal(requestedUrl, "/api/v1/cases/case-001/mask");
  assert.equal(result.byteLength, 3);
});
