import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false
  }
})

