@echo off
setlocal
title Photographic Studio By Bill - Calibrate a camera
rem ---------------------------------------------------------------------------
rem  Calibrate Camera.bat  (Phase 2 engine; the friendly wizard comes in Phase 7)
rem    Drag the folder of calibration frames onto this file:
rem      Part A - dark frames (body cap on) at every ISO, up and back down
rem      Part B - flat frames in PAIRS (same ISO and shutter speed), evenly lit
rem    Frames are sorted automatically; nothing needs renaming.
rem    The profile is saved privately in calibration_local\profiles (never shared
rem    unless you choose to). Hot-pixel maps stay on this PC.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
if "%~1"=="" goto :help
"%PY%" -m studio calibrate %*
if errorlevel 1 goto :fail
echo.
pause
endlocal
exit /b 0
:help
echo.
echo  Drag the folder (or files) of calibration frames onto this .bat file.
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
