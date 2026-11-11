import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from https://kushsharawat.github.io/churn-prediction-app/ in production
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/churn-prediction-app/' : '/',
}))
