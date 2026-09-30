import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // 只读取服务端环境变量（不带 VITE_ 前缀）：密钥留在开发服务器进程里，
  // 不会像 VITE_* 那样被内联进浏览器产物。
  const env = loadEnv(mode, process.cwd(), "");
  const apiKey = (env.BTA_API_KEY || "").trim();

  const apiProxy = {
    "/api": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
      headers: apiKey ? { "X-API-Key": apiKey } : {},
    },
  };

  return {
    cacheDir: ".vite-cache",
    plugins: [react()],
    server: {
      // 仅监听本机：API 密钥由代理在服务端注入，浏览器端不持有任何凭据。
      host: "127.0.0.1",
      port: 5173,
      proxy: apiProxy,
    },
    preview: {
      host: "127.0.0.1",
      port: 5173,
      proxy: apiProxy,
    },
  };
});
