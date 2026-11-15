@echo off
setlocal
title Photographic Studio By Bill - Denoise benchmark
rem ---------------------------------------------------------------------------
rem  Run Benchmark.bat - scores every denoiser candidate on synthetic noise
rem  measured from your own high-ISO raws, added to your clean low-ISO raws
rem  (so the right answer is known). Uses the raws in this folder and in
rem  "high iso raw". The report opens in your browser; it stays on this PC.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
"%PY%" "%~dp0tools\bench.py" --auto %*
if errorlevel 1 goto :fail
echo.
echo  Finished. The report is in outputs\bench
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
