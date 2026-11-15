@echo off
setlocal EnableExtensions
title Photographic Studio By Bill - Make a release
rem ===========================================================================
rem  Release.bat - makes a new version everyone can download and update to.
rem  Run push.bat first (so your changes are on GitHub), then this.
rem    1. asks for the new version number (it shows the current one)
rem    2. writes it into the Studio, commits it, and tags it (v1.2.3)
rem    3. sends it to GitHub. GitHub then builds the MSI on its own Windows
rem       computers and publishes the release, with Install-Photographic-Studio.bat
rem       beside it (about 10-20 minutes). Copies with "Check for updates" on
rem       will find it.
rem  The first time, it also puts GitHub's build recipe in place (tools\github_release.yml
rem  copied to .github\workflows\release.yml) and sends it with the release.
rem  Every run is logged in logs\release.log (never on C:).
rem
rem  push.bat runs this by itself after every push ("/auto"): no questions, the
rem  last number goes up by one (0.2.0 to 0.2.1), and no pause at the end. Run
rem  Release.bat yourself only when you want a bigger jump (0.2.1 to 0.3.0).
rem ===========================================================================
set "AUTO="
if /i "%~1"=="/auto" set "AUTO=1"
cd /d "%~dp0"
if not exist "logs\" mkdir "logs"
set "LOG=%~dp0logs\release.log"
set "ACTIONS=https://github.com/photogbill/Photographic_Studio_By_Bill/actions"
echo [%DATE% %TIME%] Release.bat started %1 >> "%LOG%"
if defined AUTO goto :banner_done
echo.
echo  Photographic Studio By Bill - make a release
echo  ============================================
echo.
:banner_done
where git >nul 2>&1
if errorlevel 1 (
  echo  Git is not installed, or Windows can't find it.
  goto :end
)
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo  This folder is not a git repository.
  goto :end
)
set "DIRTY="
for /f "usebackq delims=" %%L in (`git status --porcelain`) do set "DIRTY=1"
if not defined DIRTY goto :clean
echo  You have changes that are not on GitHub yet. Run push.bat first,
echo  then run Release.bat again.
echo [%DATE% %TIME%] stopped: uncommitted changes >> "%LOG%"
goto :end

:clean
rem  GitHub's build recipe lives in tools\github_release.yml; GitHub only reads it from
rem  .github\workflows, so it is copied there (and sent with the release) when missing or changed.
set "WF=%~dp0.github\workflows\release.yml"
set "WF_NEW="
fc /b "%~dp0tools\github_release.yml" "%WF%" >nul 2>nul
if not errorlevel 1 goto :wf_ok
if not exist "%~dp0.github\workflows\" mkdir "%~dp0.github\workflows"
copy /y "%~dp0tools\github_release.yml" "%WF%" >nul
echo  Added GitHub's build recipe for the MSI (.github\workflows\release.yml).
echo [%DATE% %TIME%] workflow copied into .github\workflows >> "%LOG%"
set "WF_NEW=1"
:wf_ok
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :end
set "CUR="
for /f "usebackq delims=" %%V in (`call "%PY%" "%~dp0tools\set_version.py" --show`) do set "CUR=%%V"
if not defined AUTO goto :ask_version
set "NEWV="
for /f "usebackq delims=" %%V in (`call "%PY%" "%~dp0tools\set_version.py" --bump`) do set "NEWV=%%V"
if not defined NEWV goto :git_fail
if "%NEWV:~0,7%"=="PROBLEM" goto :git_fail
goto :make
:ask_version
echo  The Studio is at version %CUR%.
echo.
echo  Type the new version number, three numbers like 0.2.0
echo    small fix: raise the last number (0.1.0 to 0.1.1)
echo    new features: raise the middle one (0.1.1 to 0.2.0)
echo  Press Enter alone to stop.
set "NEWV="
set /p "NEWV=  > "
if not defined NEWV goto :end
set "NEWV=%NEWV:v=%"
set "NEWV=%NEWV:V=%"
"%PY%" "%~dp0tools\set_version.py" %NEWV% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo.
  echo  "%NEWV%" can't be used: it must be three numbers like 0.2.0, higher than %CUR%.
  goto :end
)
:make
echo.
echo  Making update %NEWV% (was %CUR%)...
git add studio/__init__.py pyproject.toml >> "%LOG%" 2>&1
if defined WF_NEW git add .github/workflows/release.yml >> "%LOG%" 2>&1
git commit -m "Release %NEWV%" >> "%LOG%" 2>&1
if errorlevel 1 goto :git_fail
git tag -a "v%NEWV%" -m "Photographic Studio %NEWV%" >> "%LOG%" 2>&1
if errorlevel 1 goto :git_fail
echo  Sending it to GitHub...
git push origin HEAD >> "%LOG%" 2>&1
if errorlevel 1 goto :push_fail
git push origin "v%NEWV%" >> "%LOG%" 2>&1
if errorlevel 1 goto :push_fail
echo [%DATE% %TIME%] release %NEWV% sent >> "%LOG%"
echo.
echo  Done: update %NEWV% is on its way. GitHub builds the installer now (about
echo  10-20 minutes); copies with "Check for updates" on will pick it up after that.
echo  Watch it here (green tick = published, red cross = send Claude the error):
echo     %ACTIONS%
if defined AUTO goto :done_auto
start "" "%ACTIONS%"
goto :end

:git_fail
echo.
echo  PROBLEM: git could not make the release commit or tag. Details: logs\release.log
goto :end
:push_fail
echo.
echo  PROBLEM: sending to GitHub did not work. Check the internet connection and
echo  run Release.bat again with the next number. Details: logs\release.log
goto :end

:end
if defined AUTO (
  endlocal
  exit /b 1
)
echo.
pause
endlocal
exit /b 0

:done_auto
endlocal
exit /b 0
