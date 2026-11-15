@echo off
setlocal
title Photographic Studio By Bill
rem ---------------------------------------------------------------------------
rem  Start Studio.bat - opens Photographic Studio By Bill.
rem  The first run installs the interface package (PySide6) into this folder.
rem  Problems are written to logs\studio.log (Help > Copy diagnostic report).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
rem  An update downloaded by the Studio (Help > Check for updates) is put in place
rem  here, before the Studio starts, by a helper that runs from a copy (so this
rem  file can be replaced too). No "call": the helper starts the Studio itself.
if /i "%~1"=="--no-update" goto :start
if not exist "%~dp0_update\ready.txt" goto :start
if not exist "%~dp0tools\apply_update.bat" goto :start
copy /y "%~dp0tools\apply_update.bat" "%~dp0_update\apply_update.bat" >nul
"%~dp0_update\apply_update.bat" "%~dp0."
:start
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
set "PYW=%PY:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" -m studio.app
endlocal
exit /b 0
:fail
echo.
echo  The Studio could not start. Details are in logs\setup.log and logs\studio.log
echo.
pause
endlocal
exit /b 1
