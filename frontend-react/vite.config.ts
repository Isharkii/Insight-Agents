import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = (
    env.VITE_BACKEND_PROXY_TARGET || "http://localhost:8000"
  )
    .trim()
    .replace(/\/+$/, "");

  return {
    base: "/",
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: 3000,
      proxy: {
        "/api/business-intelligence": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/analyze": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/health": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/export": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/upload-csv": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/compute-kpis": {
          target: proxyTarget,
          changeOrigin: true,
        },
        "/clients": {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
