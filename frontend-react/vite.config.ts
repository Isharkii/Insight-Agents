import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
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
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/analyze": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/export": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/upload-csv": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/compute-kpis": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/clients": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
