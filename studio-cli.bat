@echo off
rem ---------------------------------------------------------------------------
rem  studio-cli.bat - the Studio from the command line, for other programs
rem  (the RPG Forge, Claude, your own scripts). Not something to double-click.
rem    studio-cli.bat schema --json          every setting and command, as JSON
rem    studio-cli.bat edit PHOTO --set light.shadows=30 --out FOLDER --json
rem  Everything is in docs\CLI.md. With --json the result is one JSON object on
rem  standard output; set-up messages go to standard error. The folder you run it
rem  from stays the current folder, so relative paths mean what you expect.
rem  Exit code: 0 worked, 1 something failed, 2 the command line was wrong,
rem  3 the Studio own Python could not be set up.
rem ---------------------------------------------------------------------------
setlocal
call "%~dp0tools\ensure_env.bat" 1>&2
if errorlevel 1 exit /b 3
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%PY%" -m studio %*
exit /b %errorlevel%
