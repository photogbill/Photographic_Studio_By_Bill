@echo off
setlocal
title Photographic Studio By Bill - Convert Models
rem ---------------------------------------------------------------------------
rem  Convert Models.bat - turns downloaded models into the Studio's own format,
rem  on this PC. Put the downloads in the models folder first:
rem    model.onnx                    Depth Anything V2 Small (depth, Masks > Depth)
rem    sam2.1_hiera_tiny_*.zip       SAM 2.1 Tiny  (click-to-select, fast)
rem    sam2.1_hiera_large_*.zip      SAM 2.1 Large (click-to-select, finest edges)
rem    lama_fp32.onnx                LaMa (Retouch > Fill; huggingface.co/Carve/LaMa-ONNX)
rem  Each is converted once (Large takes several minutes) and checked. Run it
rem  again after adding another download; finished ones are skipped.
rem  Add  --force  to convert everything again.
rem  Work files go in cache\ and are removed; nothing is written to C:\.
rem  Full log: logs\convert_models.log
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail_env
if not exist "%~dp0cache\tmp" mkdir "%~dp0cache\tmp"
set "TEMP=%~dp0cache\tmp"
set "TMP=%~dp0cache\tmp"
set "CLOG=%~dp0logs\convert_models.log"
echo [%DATE% %TIME%] Convert Models.bat started >> "%CLOG%"

"%PY%" -c "import onnx" >nul 2>nul
if errorlevel 1 (
  echo  Adding the model converter's tools ^(onnx, one time only^)...
  "%PY%" -m pip install onnx >> "%CLOG%" 2>&1
)
"%PY%" -c "import onnxsim" >nul 2>nul
if errorlevel 1 "%PY%" -c "import onnxslim" >nul 2>nul
if errorlevel 1 (
  echo  Adding the graph folder ^(onnxsim, one time only^)...
  "%PY%" -m pip install onnxsim >> "%CLOG%" 2>&1
  "%PY%" -c "import onnxsim" >nul 2>nul
  if errorlevel 1 "%PY%" -m pip install onnxslim >> "%CLOG%" 2>&1
)
"%PY%" -c "import onnxruntime" >nul 2>nul
if errorlevel 1 (
  echo  Adding ONNX Runtime, used only here to check the converted models ^(one time only^)...
  "%PY%" -m pip install onnxruntime >> "%CLOG%" 2>&1
)
"%PY%" -c "import onnx" >nul 2>nul
if errorlevel 1 goto :fail_tools

echo.
"%PY%" tools\convert_models.py %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
  powershell -NoProfile -Command "Write-Host '  MODELS READY' -ForegroundColor Black -BackgroundColor Green"
) else (
  powershell -NoProfile -Command "Write-Host '  SOMETHING DID NOT FINISH - see logs\convert_models.log' -ForegroundColor White -BackgroundColor Red"
)
echo.
pause
endlocal
exit /b %RC%

:fail_tools
echo.
echo  The converter's tools could not be installed. Check the internet connection
echo  and run this again. Details: logs\convert_models.log
pause
endlocal
exit /b 1

:fail_env
echo.
echo  The Python environment could not be prepared. Details: logs\setup.log
pause
endlocal
exit /b 1
