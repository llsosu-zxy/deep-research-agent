@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Create it with: python -m venv .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" ui\gradio_app.py
