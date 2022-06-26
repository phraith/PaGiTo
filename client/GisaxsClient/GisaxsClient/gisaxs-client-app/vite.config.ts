import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import visualizer from 'rollup-plugin-visualizer'
import mkcert from 'vite-plugin-mkcert'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), mkcert()],
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        sourcemap: true
      },
      plugins: [
        visualizer()
      ]
    }
  },
  server: {
    // https: true,
    proxy: {
      '/api': { target: 'https://localhost:9999', secure: false },
      '/message': {
        target: 'ws://localhost:9998',
        ws: true
      }
    }
  }
})
