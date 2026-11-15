@echo off
setlocal
title Photographic Studio By Bill - Install
rem ---------------------------------------------------------------------------
rem  Install.bat - double-click once. Builds the private Python environment
rem  inside this folder and checks that everything works.
rem  Nothing is installed anywhere else on the PC, and nothing goes to C:.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
echo.
echo  Photographic Studio By Bill - install
echo  =====================================
echo.
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
echo.
echo  Checking the engine...
"%PY%" -m studio version
if errorlevel 1 goto :fail
"%PY%" -c "import rawpy; print('  LibRaw', '.'.join(map(str, rawpy.libraw_version)), '- reads CR2 and CR3')"
if errorlevel 1 goto :fail
echo.
echo  All set. Next:
echo    - drag raw files onto "Develop Raws.bat" to develop them
echo    - double-click "Run Tests.bat" to run the self-tests
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo  Install did not finish. The details are in logs\setup.log and logs\studio.log.
echo.
pause
endlocal
exit /b 1
