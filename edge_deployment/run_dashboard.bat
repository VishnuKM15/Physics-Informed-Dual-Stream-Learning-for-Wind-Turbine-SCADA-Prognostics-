@echo off
title CARE Wind Turbine Prognostics Dashboard
cd /d "%~dp0"

echo ======================================================================
echo   CARE WIND TURBINE PROGNOSTICS DASHBOARD
echo   Fraunhofer IEE Benchmark - Physics-Informed Dual-Stream Edge Model
echo ======================================================================

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else if exist "..\.venv\Scripts\python.exe" (
    set PYTHON_CMD=..\.venv\Scripts\python.exe
)

echo [*] Using Python: %PYTHON_CMD%
echo [*] Starting Edge Dashboard Server at http://localhost:5000 ...

start "" http://localhost:5000
%PYTHON_CMD% app.py

pause
