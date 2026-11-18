@echo off
setlocal
title Photographic Studio By Bill - Clean on your camera's real noise
rem ---------------------------------------------------------------------------
rem  Run Noise Benchmark.bat - scores every Clean method on YOUR camera's noise.
rem    Pictures with a known right answer (charts, and your clean photos once
rem    the Lab has prepared some) are given the camera's real noise: photon
rem    noise by its measured gain plus patches of its own dark frames. Each
rem    method cleans them, with and without the detail guard, and is scored on
rem    noise left, texture kept, edges kept, colour blotches and whether what
rem    it removed was only noise.
rem    Needs a Full profile (dark frames and flat pairs) and, for real noise,
rem    Deep calibration on the camera's card in the Cameras page.
rem    Add --quick for a few minutes instead of many.
rem  The report opens in your browser; it stays on this PC (outputs\noisebench).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
"%PY%" "%~dp0tools\noise_bench.py" %*
if errorlevel 1 goto :fail
echo.
echo  Finished. The report is in outputs\noisebench
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
