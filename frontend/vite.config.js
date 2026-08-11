import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',  // Force IPv4 — avoids IPv6/IPv4 mismatch with FastAPI backend
    proxy: {
      // All /api/* calls proxied to FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // Direct backend routes (health, transcribe, etc.)
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/upload-audio': { target: 'http://localhost:8000', changeOrigin: true },
      '/transcribe': { target: 'http://localhost:8000', changeOrigin: true },
      '/diarize': { target: 'http://localhost:8000', changeOrigin: true },
      '/analyze-stress': { target: 'http://localhost:8000', changeOrigin: true },
      '/lap-times': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
