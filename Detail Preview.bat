@echo off
setlocal
title Photographic Studio By Bill - Detail Preview
rem ---------------------------------------------------------------------------
rem  Detail Preview.bat
rem    Shows microcontrast ("more shades of shadow") and Recover lens sharpness
rem    on your own raws: a before/after wipe of each photo and 100% crops,
rem    next to your DxO export when NAME_DxO.jpg sits beside the raw.
rem    Drag raw files or a folder onto this file, or double-click it to use
rem    the Voigtlander folder (the first 6 photos).
rem  Results go to outputs\detail_preview in this folder (never to C:).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
echo.
echo  Building the detail preview (about a minute per photo)...
echo.
"%PY%" -m studio detail-preview %*
if errorlevel 1 goto :fail
echo.
echo  Finished. The page opened in your browser (outputs\detail_preview).
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
