@echo off
setlocal
title Photographic Studio By Bill - Blind test vs DxO
rem ---------------------------------------------------------------------------
rem  Blind Test vs DxO.bat  (Phase 3 acceptance tool)
rem    Double-click: uses the Voigtlander folder (raw + NAME_DxO.jpg pairs).
rem    Or drag any folder that holds raws with their DxO exports beside them.
rem    A page opens: pick the better of two unlabelled 100 percent crops, ten times.
rem    It tells you at the end how often you picked the Studio.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
"%PY%" -m studio blindtest %*
if errorlevel 1 goto :fail
echo.
pause
endlocal
exit /b 0
:fail
echo.
echo  Something went wrong. Details are in logs\studio.log
echo.
pause
endlocal
exit /b 1
