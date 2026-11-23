@echo off
setlocal
title Photographic Studio By Bill - extension file for Claude
cd /d "%~dp0"
rem Make Claude Extension.bat - the other way to connect the Studio to the Claude desktop app.
rem Use it when Claude has no Studio tools after "Connect Studio to Claude.bat" and a full restart.
rem Makes extension\photographic-studio.mcpb (it only says how to start the Studio in this folder),
rem opens the folder with the file selected, and says where the Install button is in Claude.
rem Changes nothing outside this folder. Safe to run again: the file is made again.
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail
"%PY%" "%~dp0tools\make_extension.py"
if errorlevel 1 goto :fail
explorer /select,"%~dp0extension\photographic-studio.mcpb"
echo.
pause
exit /b 0
:fail
echo.
echo  The extension file was not made - scroll up for the reason.
pause
exit /b 1
