@echo off
rem ---------------------------------------------------------------------------
rem  tools\ensure_env.bat - used by the other .bat files. Do not run it directly.
rem  Makes sure the app's private Python exists and has every package in
rem  requirements.txt. Sets PY to its python.exe. Two kinds, both in the app folder:
rem    python\   a private copy of Python itself (made by the installer, or here
rem              when the PC has no Python): nothing to install on Windows first
rem    .venv\    a virtual environment of the PC's own Python (developer copies)
rem  Nothing is installed outside the app folder. pip's cache is turned off so
rem  nothing lands in AppData either.
rem ---------------------------------------------------------------------------
set "APP=%~dp0.."
for %%I in ("%APP%") do set "APP=%%~fI"
set "VENV=%APP%\.venv"
set "EMB=%APP%\python"
set "PY=%VENV%\Scripts\python.exe"
set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
if not exist "%APP%\logs" mkdir "%APP%\logs"
set "SETUPLOG=%APP%\logs\setup.log"

if exist "%EMB%\python.exe" set "PY=%EMB%\python.exe"
if exist "%PY%" goto :have_python
if "%STUDIO_PRIVATE_PYTHON%"=="1" goto :private_python
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
if not exist "%PY%" goto :private_python
rem  Too old a Python (the packages need 3.10 or newer): use a private copy instead.
"%PY%" -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul
if not errorlevel 1 goto :have_python
echo [%DATE% %TIME%] the PC's Python is older than 3.10: using a private copy >> "%SETUPLOG%"
rmdir /s /q "%VENV%" >nul 2>nul
set "PY=%VENV%\Scripts\python.exe"

:private_python
call :get_python
if errorlevel 1 exit /b 1
set "PY=%EMB%\python.exe"

:have_python
rem  requirements.txt is checked against a copy kept beside the Python, so an
rem  update that needs a new package installs it on the next start.
set "MARKDIR=%VENV%"
if /i "%PY%"=="%EMB%\python.exe" set "MARKDIR=%EMB%"
"%PY%" -c "import numpy, scipy, rawpy, tifffile, PIL, pytest, PySide6" >nul 2>nul
if errorlevel 1 goto :install_packages
fc /b "%APP%\requirements.txt" "%MARKDIR%\requirements.installed" >nul 2>nul
if not errorlevel 1 goto :optional
:install_packages
echo  Installing the packages the Studio needs (numpy, scipy, rawpy, tifffile, Pillow, PySide6, pytest)...
echo  This takes a minute or two the first time.
echo [%DATE% %TIME%] pip install -r requirements.txt >> "%SETUPLOG%"
"%PY%" -m pip install --upgrade pip >> "%SETUPLOG%" 2>&1
"%PY%" -m pip install -r "%APP%\requirements.txt" >> "%SETUPLOG%" 2>&1
if not errorlevel 1 copy /y "%APP%\requirements.txt" "%MARKDIR%\requirements.installed" >nul
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
rem  inside .venv (or python\) stops it retrying on every start (delete it to try again).
if exist "%MARKDIR%\lensfun_tried.txt" goto :astro
"%PY%" -c "import lensfunpy" >nul 2>nul
if not errorlevel 1 goto :astro
echo  Adding the lens database (lensfun, optional, one time only)...
echo [%DATE% %TIME%] pip install lensfunpy (optional) >> "%SETUPLOG%"
"%PY%" -m pip install lensfunpy >> "%SETUPLOG%" 2>&1
echo tried %DATE% %TIME% > "%MARKDIR%\lensfun_tried.txt"
"%PY%" -c "import lensfunpy" >nul 2>nul
if errorlevel 1 (
  echo  The lens database could not be added. Everything else works; modern-lens profiles are unavailable.
) else (
  echo  Lens database added.
)

:astro
rem  astropy + photutils (BSD-3): FITS files and the astronomers' star finder. Optional.
if exist "%MARKDIR%\astro_tried.txt" exit /b 0
"%PY%" -c "import astropy.io.fits, photutils.detection" >nul 2>nul
if not errorlevel 1 exit /b 0
echo  Adding the astronomy tools (astropy, optional, one time only)...
echo [%DATE% %TIME%] pip install astropy photutils (optional) >> "%SETUPLOG%"
"%PY%" -m pip install astropy photutils >> "%SETUPLOG%" 2>&1
echo tried %DATE% %TIME% > "%MARKDIR%\astro_tried.txt"
"%PY%" -c "import astropy.io.fits" >nul 2>nul
if errorlevel 1 (
  echo  The astronomy tools could not be added. Everything else works; FITS files are unavailable.
) else (
  echo  Astronomy tools added.
)
exit /b 0

rem ---------------------------------------------------------------------------
rem  :get_python - a private copy of Python (python.org's official "embeddable"
rem  package: no installer, nothing written to Windows) in the app's python\ folder.
:get_python
echo  Downloading a private copy of Python into this folder (about 11 MB, one time only)...
echo [%DATE% %TIME%] fetching the embeddable Python >> "%SETUPLOG%"
if not exist "%EMB%" mkdir "%EMB%"
for %%V in (3.13.12 3.13.11 3.13.9 3.12.10) do if not exist "%EMB%\python.exe" call :fetch_python %%V
if exist "%EMB%\python.exe" goto :got_python
echo.
echo  PROBLEM: Python could not be downloaded. Check the internet connection and
echo  run this again. Details: logs\setup.log
exit /b 1
:got_python
rem  Let this Python see its own packages (Lib\site-packages) and the Studio (the folder above it).
set "PZIP="
for %%Z in ("%EMB%\python3*.zip") do set "PZIP=%%~nxZ"
for %%F in ("%EMB%\python3*._pth") do (
  > "%%~fF" echo %PZIP%
  >> "%%~fF" echo .
  >> "%%~fF" echo ..
  >> "%%~fF" echo import site
)
if exist "%EMB%\Lib\site-packages\pip\" exit /b 0
echo  Adding pip, the package installer, to it...
call :download "https://bootstrap.pypa.io/get-pip.py" "%EMB%\get-pip.py"
if errorlevel 1 goto :no_pip
"%EMB%\python.exe" "%EMB%\get-pip.py" --no-warn-script-location >> "%SETUPLOG%" 2>&1
del /q "%EMB%\get-pip.py" 2>nul
"%EMB%\python.exe" -m pip --version >> "%SETUPLOG%" 2>&1
if not errorlevel 1 exit /b 0
:no_pip
echo.
echo  PROBLEM: pip could not be added. Check the internet connection and run this
echo  again. Details: logs\setup.log
exit /b 1

:fetch_python
set "PYV=%~1"
echo [%DATE% %TIME%] trying Python %PYV% >> "%SETUPLOG%"
call :download "https://www.python.org/ftp/python/%PYV%/python-%PYV%-embed-amd64.zip" "%EMB%\python.zip"
if errorlevel 1 exit /b 0
set "UZ_ZIP=%EMB%\python.zip"
tar -xf "%UZ_ZIP%" -C "%EMB%" >> "%SETUPLOG%" 2>&1
if errorlevel 1 powershell -NoProfile -Command "Expand-Archive -Force -LiteralPath $env:UZ_ZIP -DestinationPath $env:EMB" >> "%SETUPLOG%" 2>&1
del /q "%EMB%\python.zip" 2>nul
if exist "%EMB%\python.exe" echo [%DATE% %TIME%] Python %PYV% unpacked >> "%SETUPLOG%"
exit /b 0

rem  :download URL FILE - curl.exe (in Windows 10 and 11), else PowerShell.
:download
set "DL_URL=%~1"
set "DL_OUT=%~2"
if exist "%DL_OUT%" del /q "%DL_OUT%"
where curl.exe >nul 2>nul
if errorlevel 1 goto :download_ps
curl.exe -L --fail --retry 3 --retry-delay 2 -s -S -o "%DL_OUT%" "%DL_URL%" >> "%SETUPLOG%" 2>&1
if not errorlevel 1 if exist "%DL_OUT%" exit /b 0
:download_ps
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -UseBasicParsing -Uri $env:DL_URL -OutFile $env:DL_OUT; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }" >> "%SETUPLOG%" 2>&1
if errorlevel 1 exit /b 1
if not exist "%DL_OUT%" exit /b 1
exit /b 0
