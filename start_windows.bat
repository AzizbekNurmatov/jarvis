@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 not found. Install Python 3.11 and try again.
    pause
    exit /b 1
  )
)
echo Installing or checking dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Review the error above.
  pause
  exit /b 1
)
echo Starting AURA...
.venv\Scripts\python.exe main.py
if errorlevel 1 pause
