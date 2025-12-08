import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  root: "./client",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    minify: "esbuild", // Faster than terser
    sourcemap: false, // Disable sourcemaps in production for faster builds
    chunkSizeWarningLimit: 1000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./client/src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
