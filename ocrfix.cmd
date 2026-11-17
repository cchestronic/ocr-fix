@echo off
rem ocrfix launcher for Windows: ./ocrfix <command> ... (PowerShell) or ocrfix <command> ... (cmd)
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
