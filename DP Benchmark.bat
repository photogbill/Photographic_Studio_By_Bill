@echo off
setlocal
title Photographic Studio By Bill - Dual Pixel speed test
rem ---------------------------------------------------------------------------
rem  DP Benchmark.bat - how many seconds a Dual Pixel frame of a focus stack
rem    takes, step by step, the old way and the new way, on your own frames.
rem    Drag a folder of Dual Pixel RAWs (one stack is enough) onto this file.
rem    It times six frames. Nothing is copied; the numbers stay on this PC
rem    (outputs\dp_bench).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
if "%~1"=="" (
  echo.
  echo  Drag a folder of Dual Pixel RAW files onto this file.
  echo.
  pause
  exit /b 1
)
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
"%PY%" "%~dp0tools\dp_bench.py" %*
if errorlevel 1 goto :fail
echo.
echo  Finished. The numbers are in outputs\dp_bench
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
