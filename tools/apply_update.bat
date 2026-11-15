@echo off
setlocal EnableExtensions
title Photographic Studio By Bill - updating
rem ---------------------------------------------------------------------------
rem  tools\apply_update.bat - used by "Start Studio.bat". Do not run it directly.
rem  Puts an update the Studio downloaded (_update\pending) in place, then starts
rem  the Studio. It runs from a copy in _update\, so every file of the app,
rem  this one and "Start Studio.bat" included, can be replaced safely.
rem  Settings, calibrations, logs and photos are never touched: the update holds
rem  only software and shared profiles, and nothing is deleted.
rem  Developer copies (a .git folder) are never overwritten.
rem ---------------------------------------------------------------------------
set "APP=%~1"
for %%I in ("%APP%") do set "APP=%%~fI"
if not exist "%APP%\logs" mkdir "%APP%\logs"
set "LOG=%APP%\logs\update.log"
set "NEWV=?"
if exist "%APP%\_update\ready.txt" set /p NEWV=<"%APP%\_update\ready.txt"
set "UPD="
for /d %%D in ("%APP%\_update\pending\*") do set "UPD=%%~fD"
if exist "%APP%\.git" goto :done
if not defined UPD goto :done
if not exist "%UPD%\studio\__init__.py" goto :done
echo.
echo  Installing Photographic Studio %NEWV%...
echo [%DATE% %TIME%] applying update %NEWV% from %UPD% >> "%LOG%"
rem  give a Studio that is just closing a moment to finish
timeout /t 2 /nobreak >nul
robocopy "%UPD%" "%APP%" /E /R:3 /W:2 /NFL /NDL /NJH /NJS /NP >> "%LOG%" 2>&1
if errorlevel 8 goto :failed
>"%APP%\.installed_version" echo %NEWV%
echo [%DATE% %TIME%] update %NEWV% is in place >> "%LOG%"
goto :done
:failed
echo  The update could not be put in place; the current version starts instead.
echo  Details: logs\update.log
echo [%DATE% %TIME%] update %NEWV% failed (robocopy) >> "%LOG%"
timeout /t 6 >nul
:done
rmdir /s /q "%APP%\_update\pending" 2>nul
del /q "%APP%\_update\ready.txt" 2>nul
endlocal & "%APP%\Start Studio.bat" --no-update
