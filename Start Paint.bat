@echo off
setlocal
title Photographic Studio By Bill - Paint
rem ---------------------------------------------------------------------------
rem  Start Paint.bat - opens Photographic Studio By Bill straight into Paint.
rem  First-run setup is the same as Start Studio.bat (tools\ensure_env.bat).
rem  If an update is waiting, Start Studio.bat puts it in place and opens the
rem  Studio; choose Paint in the sidebar that one time.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
if exist "%~dp0_update\ready.txt" (
    call "%~dp0Start Studio.bat"
    endlocal
    exit /b 0
)
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
set "PYW=%PY:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" -m studio.app --paint
endlocal
exit /b 0
:fail
echo.
echo  Paint could not start. Details are in logs\setup.log and logs\studio.log
echo.
pause
endlocal
exit /b 1
