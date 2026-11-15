@echo off
setlocal
title Photographic Studio By Bill - Self-tests
rem ---------------------------------------------------------------------------
rem  Run Tests.bat - runs every self-test and prints a green / red summary.
rem  Tests use synthetic pictures made on the fly. If raw files sit in this
rem  folder, a few extra tests also develop them. Nothing is uploaded.
rem  Full output: logs\tests.log
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail_env

echo.
echo  Running the self-tests (this can take a few minutes)...
echo.
"%PY%" -m pytest -q -p no:cacheprovider tests > "%~dp0logs\tests.log" 2>&1
set "RC=%errorlevel%"
powershell -NoProfile -Command "Get-Content -LiteralPath '%~dp0logs\tests.log' -Tail 25"
echo.
if "%RC%"=="0" (
  powershell -NoProfile -Command "Write-Host '  ALL TESTS PASSED' -ForegroundColor Black -BackgroundColor Green"
) else (
  powershell -NoProfile -Command "Write-Host '  SOME TESTS FAILED - see logs\tests.log' -ForegroundColor White -BackgroundColor Red"
)
echo.
pause
endlocal
exit /b %RC%

:fail_env
echo.
echo  The Python environment could not be prepared. Details: logs\setup.log
pause
endlocal
exit /b 1
