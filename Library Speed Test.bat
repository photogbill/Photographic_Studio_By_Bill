@echo off
setlocal
title Photographic Studio By Bill - Library speed test
rem ---------------------------------------------------------------------------
rem  Library Speed Test.bat - how fast a big folder opens in the Library.
rem    Drag a folder of raws onto this file (e.g. one on your USB archive
rem    drive) to measure that folder, or double-click it to use a 2,000-photo
rem    test folder made from the sample raws (hard links: no extra disk space).
rem    Nothing in your folder is changed. A temporary thumbnail cache is used,
rem    so the first visit is measured cold. Results: outputs\bench_library.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
echo.
echo  Measuring (a minute or two)...
echo.
"%PY%" "%~dp0tools\bench_library.py" %*
if errorlevel 1 goto :fail
echo.
echo  Finished. The numbers are in outputs\bench_library\results.txt
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
