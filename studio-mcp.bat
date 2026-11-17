@echo off
rem ---------------------------------------------------------------------------
rem  studio-mcp.bat - starts the Studio as an MCP server, so an AI model can use
rem  its tools. Not something to double-click: an MCP client (the Claude desktop
rem  app, or any other) starts it and talks to it. To set Claude up, double-click
rem  "Connect Studio to Claude.bat" instead. Details: docs\CLI.md, section MCP.
rem  Standard output carries the protocol only; everything else goes to standard
rem  error and to logs\studio.log.
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0tools\ensure_env.bat" 1>&2
if errorlevel 1 exit /b 3
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%PY%" -m studio.mcp_server
exit /b %errorlevel%
