@echo off
rem ---------------------------------------------------------------------------
rem  tools\ensure_env.bat - used by the other .bat files. Do not run it directly.
rem  Makes sure the private Python environment (.venv in the app folder) exists
rem  and has every package in requirements.txt. Sets PY to its python.exe.
rem  Nothing is installed outside the app folder. pip's cache is turned off so
rem  nothing lands in AppData either.
rem ---------------------------------------------------------------------------
set "APP=%~dp0.."
for %%I in ("%APP%") do set "APP=%%~fI"
set "VENV=%APP%\.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
if not exist "%APP%\logs" mkdir "%APP%\logs"
set "SETUPLOG=%APP%\logs\setup.log"

if exist "%PY%" goto :have_python
echo  First run: building a private Python environment inside this folder (one time only)...
echo [%DATE% %TIME%] creating venv >> "%SETUPLOG%"
where py >nul 2>nul
if errorlevel 1 goto :try_python
py -3.12 -m venv "%VENV%" >> "%SETUPLOG%" 2>&1
if exist "%PY%" goto :have_python
py -3 -m venv "%VENV%" >> "%SETUPLOG%" 2>&1
if exist "%PY%" goto :have_python
:try_python
python -m venv "%VENV%" >> "%SETUPLOG%" 2>&1
if exist "%PY%" goto :have_python
echo.
echo  PROBLEM: Python 3 was not found on this PC.
echo  Install Python 3.12 from python.org and tick "Add python.exe to PATH",
echo  then run this again. Details: logs\setup.log
exit /b 1

:have_python
"%PY%" -c "import numpy, scipy, rawpy, tifffile, PIL, pytest, PySide6" >nul 2>nul
if not errorlevel 1 goto :optional
echo  Installing the packages the Studio needs (numpy, scipy, rawpy, tifffile, Pillow, PySide6, pytest)...
echo  This takes a minute or two the first time.
echo [%DATE% %TIME%] pip install -r requirements.txt >> "%SETUPLOG%"
"%PY%" -m pip install --upgrade pip >> "%SETUPLOG%" 2>&1
"%PY%" -m pip install -r "%APP%\requirements.txt" >> "%SETUPLOG%" 2>&1
"%PY%" -c "import numpy, scipy, rawpy, tifffile, PIL, pytest, PySide6" >nul 2>nul
if errorlevel 1 (
  echo.
  echo  PROBLEM: the package install did not finish.
  echo  Check the internet connection and run this again. Details: logs\setup.log
  exit /b 1
)
echo  Packages installed.

:optional
rem  Optional extras: the Studio works without them. Tried once; a marker file
rem  inside .venv stops it retrying on every start (delete it to try again).
if exist "%VENV%\lensfun_tried.txt" exit /b 0
"%PY%" -c "import lensfunpy" >nul 2>nul
if not errorlevel 1 exit /b 0
echo  Adding the lens database (lensfun, optional, one time only)...
echo [%DATE% %TIME%] pip install lensfunpy (optional) >> "%SETUPLOG%"
"%PY%" -m pip install lensfunpy >> "%SETUPLOG%" 2>&1
echo tried %DATE% %TIME% > "%VENV%\lensfun_tried.txt"
"%PY%" -c "import lensfunpy" >nul 2>nul
if errorlevel 1 (
  echo  The lens database could not be added. Everything else works; modern-lens profiles are unavailable.
) else (
  echo  Lens database added.
)
exit /b 0
