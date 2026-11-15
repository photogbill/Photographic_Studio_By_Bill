@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Photographic Studio By Bill - Push to GitHub
rem ==========================================================================
rem  push.bat - sends the software and shared profiles to GitHub.
rem  Double-click it. It shows every file before anything is sent, and asks.
rem  Photos, raws, calibration data, logs and reports are never sent:
rem  .gitignore keeps them out, and this script checks again.
rem  Every run is logged in logs\ beside this file (never on C:).
rem ==========================================================================

cd /d "%~dp0"

if not exist "logs\" mkdir "logs"
set "STAMP="
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"`) do set "STAMP=%%T"
if not defined STAMP set "STAMP=run"
set "LOG=%~dp0logs\push_%STAMP%.log"
set "LIST=%~dp0logs\push_%STAMP%_files.txt"
set "MSGFILE=%~dp0logs\push_%STAMP%_message.txt"
call :log "push.bat started in %CD%"

echo.
echo  Photographic Studio By Bill - push to GitHub
echo  ============================================
echo.

where git >nul 2>&1
if errorlevel 1 (
  call :fail "Git is not installed, or Windows can't find it. Install Git for Windows from git-scm.com, then run this again."
  goto :end
)
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  call :fail "This folder is not a git repository."
  goto :end
)
for /f "usebackq delims=" %%B in (`git rev-parse --abbrev-ref HEAD`) do set "BRANCH=%%B"
call :log "Branch: !BRANCH!"

rem --- 1. Bring in anything already on GitHub, so the push can't be refused ---
echo  1. Getting any changes already on GitHub...
call :run git pull --rebase --autostash
if errorlevel 1 (
  call :run git rebase --abort
  call :fail "Could not bring in the changes from GitHub, so nothing was sent."
  goto :end
)

rem --- 2. Collect what changed (.gitignore decides what is allowed) ---
echo  2. Collecting the files that changed...
call :run git add -A
if errorlevel 1 (
  call :fail "Git could not collect the changed files."
  goto :end
)
git -c core.quotepath=off diff --cached --name-status > "%LIST%" 2>>"%LOG%"
set "AHEAD=0"
for /f "usebackq delims=" %%A in (`git rev-list --count @{u}..HEAD 2^>nul`) do set "AHEAD=%%A"
for %%S in ("%LIST%") do set "LISTSIZE=%%~zS"
if "!LISTSIZE!"=="0" (
  if "!AHEAD!"=="0" (
    echo.
    echo  Nothing has changed since the last push. Nothing to send.
    call :log "Nothing to commit and nothing waiting to push."
    goto :end
  )
  echo.
  echo  No new changes, but !AHEAD! saved change^(s^) from before have not reached GitHub yet.
  call :log "Nothing new to commit; !AHEAD! commit(s) waiting to push."
  goto :push
)

rem --- 3. Safety check: no pictures, raws or big files, whatever .gitignore says ---
echo  3. Safety check: no photos, raws or big files...
set "BAD=0"
for /f "usebackq delims=" %%F in (`git -c core.quotepath=off diff --cached --name-only --diff-filter=ACMR`) do (
  set "P=%%F"
  set "P=!P:/=\!"
  set "EXT="
  for %%Z in ("!P!") do set "EXT=%%~xZ"
  for %%E in (.cr2 .cr3 .crw .nef .nrw .arw .srf .sr2 .raf .orf .rw2 .pef .dng .x3f .iiq .3fr .erf .mrw .srw .rwl .kdc .dcr .mos .jpg .jpeg .tif .tiff .heic .psd .psb .pdf .npy .npz) do (
    if /i "!EXT!"=="%%E" (
      set "BAD=1"
      echo     BLOCKED, picture, raw or data file: %%F
      >>"%LOG%" echo BLOCKED type: %%F
    )
  )
  if /i "!EXT!"==".png" if /i not "!P:~0,13!"=="assets\icons\" (
    set "BAD=1"
    echo     BLOCKED, PNG outside assets\icons: %%F
    >>"%LOG%" echo BLOCKED png: %%F
  )
  set "SZ="
  for %%Z in ("!P!") do set "SZ=%%~zZ"
  if not defined SZ set "SZ=0"
  if !SZ! GTR 10485760 (
    set "BAD=1"
    echo     BLOCKED, bigger than 10 MB: %%F
    >>"%LOG%" echo BLOCKED size: %%F
  )
)
if "!BAD!"=="1" (
  call :run git reset -q
  call :fail "Some files should never go to GitHub, so nothing was sent. Move them out of this folder, or tell Claude which rule to change."
  goto :end
)

rem --- Show every file, then ask before anything is sent ---
set "COUNT=0"
for /f "usebackq delims=" %%L in ("%LIST%") do set /a COUNT+=1
echo.
echo  These !COUNT! file^(s^) will be sent. A = new, M = changed, D = deleted, R = renamed:
echo.
for /f "usebackq delims=" %%L in ("%LIST%") do echo     %%L
type "%LIST%" >> "%LOG%"
echo.
set "MSG="
set /p "MSG=  Describe this change in a few words, or just press Enter: "
if not defined MSG set "MSG=Update from push.bat"
(echo(!MSG!)> "%MSGFILE%"
>>"%LOG%" echo Commit message:
type "%MSGFILE%" >> "%LOG%"
echo.
choice /c YN /n /m "  Send these files to the public GitHub repository now? [Y/N] "
if errorlevel 2 (
  call :run git reset -q
  call :log "Cancelled by the user."
  echo.
  echo  Cancelled. Nothing was sent.
  goto :end
)

echo.
echo  4. Saving the change on this PC...
call :run git commit -F "%MSGFILE%"
if errorlevel 1 (
  call :fail "The change could not be saved. Nothing was sent."
  goto :end
)

:push
echo  5. Sending to GitHub. The first time, a sign-in window may open...
echo.
call :log "Running: git push -u origin HEAD"
git push -u origin HEAD
set "RC=!errorlevel!"
call :log "git push exit code !RC!"
if not "!RC!"=="0" (
  call :fail "The push did not go through. Your change is saved on this PC: run push.bat again to retry."
  goto :end
)
set "URL="
for /f "usebackq delims=" %%U in (`git remote get-url origin`) do set "URL=%%U"
set "URL=!URL:.git=!"
echo.
echo  Done. It's on GitHub: !URL!
call :log "Pushed to !URL!"

:end
call :log "push.bat finished"
echo.
echo  Log: %LOG%
echo.
pause
endlocal
exit /b 0

rem ---------------------------------------------------------------- helpers
:run
>>"%LOG%" echo [%TIME%] ^> %*
%* >>"%LOG%" 2>&1
set "RC=!errorlevel!"
>>"%LOG%" echo [%TIME%] exit code !RC!
exit /b !RC!

:log
>>"%LOG%" echo [%DATE% %TIME%] %~1
exit /b 0

:fail
echo.
echo  PROBLEM: %~1
call :log "PROBLEM: %~1"
echo.
echo  Last lines of the log:
powershell -NoProfile -Command "Get-Content -LiteralPath $env:LOG -Tail 15"
exit /b 1
