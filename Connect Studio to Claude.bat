@echo off
setlocal
title Photographic Studio By Bill - Connect to Claude
rem ---------------------------------------------------------------------------
rem  Connect Studio to Claude.bat
rem    Double-click to let the Claude desktop app use the Studio (MCP).
rem    It adds one entry to Claude own settings file and keeps a copy of the
rem    file as it was. Run "Disconnect" from the question below to undo it.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
echo.
echo  Photographic Studio By Bill - connect to the Claude desktop app
echo  ===============================================================
echo  This lets Claude use the Studio: develop, clean, match colours, cut out,
echo  depth maps and stacks, on the pictures you point it to.
echo.
echo    C = connect (or refresh the connection)
echo    D = disconnect
echo    Q = quit
echo.
choice /c CDQ /n /m "  Press C, D or Q: "
if errorlevel 3 goto :end
if errorlevel 2 goto :remove
"%PY%" "%~dp0tools\connect_claude.py"
goto :done
:remove
"%PY%" "%~dp0tools\connect_claude.py" --remove
:done
echo.
pause
goto :end
:fail
echo.
echo  Something went wrong setting up the Studio. Details are in logs\setup.log
echo.
pause
endlocal
exit /b 1
:end
endlocal
exit /b 0
