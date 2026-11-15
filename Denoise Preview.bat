@echo off
setlocal
title Photographic Studio By Bill - Denoise Preview (experimental)
rem ---------------------------------------------------------------------------
rem  Denoise Preview.bat  (Phase 3 groundwork, experimental)
rem    Drag high-ISO CR2 / CR3 files onto this file. A page opens with:
rem      - the camera JPEG next to the Studio render, denoised
rem      - a 100 percent crop: Studio without denoise vs Studio denoised, and Canon's own
rem    Double-click it to preview the raws in this folder and in "high iso raw".
rem  Nothing big is written: only the preview page, in outputs\develop.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
if "%~1"=="" goto :samples
"%PY%" -m studio develop %* --denoise --compare-only
if errorlevel 1 goto :fail
goto :done
:samples
"%PY%" -m studio develop "%~dp0." "%~dp0high iso raw" --denoise --compare-only
if errorlevel 1 goto :fail
:done
echo.
echo  Finished. The page is in outputs\develop (it should have opened in your browser).
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
