@echo off
setlocal
title Photographic Studio By Bill - Calibrate a camera
rem ---------------------------------------------------------------------------
rem  Calibrate Camera.bat  (the same job as the Cameras page in the Studio)
rem    Drag a folder, a card or files onto this file. One big pile is fine:
rem      dark frames (lens cap on) at every ISO, cool and warm, short and long
rem      optional: flat frames in PAIRS (same ISO and shutter speed), evenly lit
rem    Frames are measured one at a time and sorted automatically; each camera
rem    in the pile gets its own profile; a later pile adds to the earlier ones.
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
