@echo off
setlocal
title Photographic Studio By Bill - Train this camera's Clean
rem ---------------------------------------------------------------------------
rem  Train Camera Model.bat - teaches Clean your camera (BUILD_PLAN_NOISE, block F).
rem    Needs, for that camera: a Full profile (dark frames and flat pairs dropped
rem    on the Cameras page), its noise bank (Deep calibration on its card) and a
rem    folder of about 200 varied clean photos (raws at ISO 400 or lower).
rem    Start it from the camera's card in the Studio, or drag the folder of clean
rem    photos onto this file. The card's "Train a general network" starts it for
rem    every calibrated camera at once (one network, used by cameras with none of
rem    their own, and the starting point of each camera's own training).
rem  The first time it installs the training tools (PyTorch, ONNX Runtime) into
rem  the lab_packages folder beside the Studio: about 2.5 GB to download. Nothing
rem  goes outside this folder; delete lab_packages to remove them.
rem  Training runs on your graphics card, stops by itself (8 hours at most) and
rem  can be stopped with Ctrl+C and started again: it carries on where it was.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
set "LAB=%~dp0lab_packages"
set "CHECK=import sys; sys.path.append(r'%LAB%'); import torch, onnx, onnxruntime"
"%PY%" -c "%CHECK%" >nul 2>nul
if not errorlevel 1 goto :train
echo.
echo  First time: installing the training tools beside the Studio (lab_packages).
echo  About 2.5 GB to download; this can take a while. Details: logs\setup.log
echo.
if not exist "%LAB%" mkdir "%LAB%"
echo [%DATE% %TIME%] lab_packages: torch (cu124) >> "%~dp0logs\setup.log"
"%PY%" -m pip install --target "%LAB%" --no-deps torch --index-url https://download.pytorch.org/whl/cu124 >> "%~dp0logs\setup.log" 2>&1
if errorlevel 1 (
  echo  The graphics-card build did not install; trying the plain one ^(slower training^)...
  "%PY%" -m pip install --target "%LAB%" --no-deps torch >> "%~dp0logs\setup.log" 2>&1
)
echo [%DATE% %TIME%] lab_packages: onnx, onnxruntime-directml >> "%~dp0logs\setup.log"
"%PY%" -m pip install --target "%LAB%" --no-deps sympy mpmath networkx jinja2 markupsafe fsspec filelock typing_extensions onnx protobuf onnxruntime-directml coloredlogs humanfriendly flatbuffers packaging >> "%~dp0logs\setup.log" 2>&1
"%PY%" -c "%CHECK%" >nul 2>nul
if errorlevel 1 (
  echo.
  echo  PROBLEM: the training tools did not install. Check the internet connection
  echo  and run this again. Details are in logs\setup.log
  goto :fail_quiet
)
:train
echo.
if "%~1"=="" (
  "%PY%" -m studio.lab.train
) else (
  "%PY%" -m studio.lab.train --clean "%~1" %2 %3 %4 %5 %6
)
if errorlevel 1 goto :fail_quiet
echo.
echo  Finished. In the Studio: Develop, Clean, method "learned".
echo.
pause
endlocal
exit /b 0
:fail
echo.
echo  Something went wrong. Details are in logs\studio.log
:fail_quiet
echo.
pause
endlocal
exit /b 1
