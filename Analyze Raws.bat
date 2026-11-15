@echo off
setlocal
title Photographic Studio By Bill - Raw Analyzer
rem ---------------------------------------------------------------------------
rem  Drag CR2/CR3 files (or a folder) onto this file, or double-click it to
rem  analyze every raw file in this folder. The report opens in your browser.
rem  Everything (Python environment, reports, logs) stays in this folder on D:.
rem ---------------------------------------------------------------------------
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%HERE%logs" mkdir "%HERE%logs"

if exist "%PY%" goto check
echo First run: building a private Python environment in this folder (one time only)...
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -m venv "%VENV%" >> "%HERE%logs\setup.log" 2>&1
) else (
  python -m venv "%VENV%" >> "%HERE%logs\setup.log" 2>&1
)
if not exist "%PY%" (
  echo.
  echo Could not find Python 3 on this PC.
  echo Install Python 3.12 from python.org ^(tick "Add python.exe to PATH"^), then run this again.
  echo Details: logs\setup.log
  pause
  exit /b 1
)

:check
"%PY%" -c "import rawpy, numpy, exifread" >nul 2>nul
if not errorlevel 1 goto run
echo Installing rawpy, numpy and exifread into this folder's environment...
"%PY%" -m pip install --upgrade pip >> "%HERE%logs\setup.log" 2>&1
"%PY%" -m pip install rawpy numpy exifread >> "%HERE%logs\setup.log" 2>&1
"%PY%" -c "import rawpy, numpy, exifread" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Package install failed. Details: logs\setup.log
  pause
  exit /b 1
)

:run
"%PY%" "%HERE%raw_analyzer.py" %*
if errorlevel 1 (
  echo.
  echo Something went wrong. Details: logs\raw_analyzer.log
  pause
  exit /b 1
)
echo.
echo Done. The report is in the reports folder.
timeout /t 5 >nul
endlocal
