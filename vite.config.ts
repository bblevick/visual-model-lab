import { defineConfig } from "vitest/config";
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  server: {
    proxy: {
      "/trainer": { target: "http://127.0.0.1:8765", changeOrigin: false },
    },
  },
  test: { include: ["tests/**/*.test.ts"] },
});
