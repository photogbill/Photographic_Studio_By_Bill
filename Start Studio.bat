@echo off
setlocal
title Photographic Studio By Bill
rem ---------------------------------------------------------------------------
rem  Start Studio.bat - opens Photographic Studio By Bill.
rem  The first run installs the interface package (PySide6) into this folder.
rem  Problems are written to logs\studio.log (Help > Copy diagnostic report).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
set "PYW=%VENV%\Scripts\pythonw.exe"
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
