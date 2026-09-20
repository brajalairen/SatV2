/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the API lives on uvicorn; in production FastAPI serves this build from its own
// origin, so the same relative /api paths work in both cases.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: true },
  // `npm test`: unit tests for the drawing geometry and the store, in a DOM without a real map.
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
