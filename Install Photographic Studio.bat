@echo off
setlocal EnableExtensions
title Photographic Studio By Bill - Installer
rem ===========================================================================
rem  Install Photographic Studio.bat - the only file you need.
rem
rem  Double-click it. It asks where to put the Studio, then downloads the
rem  newest version from GitHub and a private copy of Python into that one
rem  folder and sets everything up. No git, no Python install, no admin rights.
rem
rem  Run it again at any time to UPDATE. Your settings, calibrations, logs and
rem  photos are kept. Nothing is installed anywhere else on the PC; the only
rem  thing outside the folder is a desktop shortcut, and only if you say yes.
rem ===========================================================================
set "REPO=photogbill/Photographic_Studio_By_Bill"
set "BRANCH=main"
if /i "%~1"=="--run" goto :run

echo.
echo  Photographic Studio By Bill - installer
echo  =======================================
echo.
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
rem  Run from inside a Studio folder: update that folder.
if exist "%HERE%\studio\__init__.py" (
  set "DEST=%HERE%"
  goto :confirm
)
rem  Otherwise: a folder on D: if the PC has one (keeps C: free), else beside this file.
set "DEST=%HERE%\Photographic Studio"
if exist "D:\" set "DEST=D:\Photographic Studio"

:confirm
echo  Install folder:  %DEST%
if exist "%DEST%\studio\__init__.py" goto :confirm_update
echo                   (a new folder)
goto :confirm_ask
:confirm_update
echo                   (the Studio is already here: it will be UPDATED,
echo                    your settings, calibrations and photos stay)
:confirm_ask
echo.
echo  Press Enter to continue, B to pick a different folder, or Q to quit.
set "ANS="
set /p "ANS=  > "
if /i "%ANS%"=="Q" goto :quit
if /i "%ANS%"=="B" goto :browse
goto :begin

:browse
set "PICK="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = 'Where should Photographic Studio go? A folder called Photographic Studio is made inside the one you pick.'; $d.ShowNewFolderButton = $true; if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"`) do set "PICK=%%P"
echo.
if not defined PICK goto :confirm
if "%PICK:~-1%"=="\" set "PICK=%PICK:~0,-1%"
set "DEST=%PICK%\Photographic Studio"
if exist "%PICK%\studio\__init__.py" set "DEST=%PICK%"
goto :confirm

:begin
if not exist "%DEST%\.git" goto :begin_ok
echo.
echo  This folder is a git copy of the Studio (a developer's own copy).
echo  The installer leaves it alone so no work is overwritten: update it with
echo  git (push.bat / git pull) instead, or pick another folder (press B).
echo.
goto :confirm
:begin_ok
if not exist "%DEST%" mkdir "%DEST%" 2>nul
if not exist "%DEST%\_installer" mkdir "%DEST%\_installer" 2>nul
if exist "%DEST%\_installer\" goto :handover
echo.
echo  PROBLEM: Windows won't let this installer write to
echo     %DEST%
echo  Pick another folder (press B), for example one on D: or in Documents.
echo.
goto :confirm

:handover
rem  The rest runs from a copy inside the install folder, so an update can
rem  safely replace this file. No "call": control passes to the copy for good.
copy /y "%~f0" "%DEST%\_installer\installer.bat" >nul
if errorlevel 1 (
  echo  PROBLEM: could not copy the installer into %DEST%\_installer
  goto :quit
)
"%DEST%\_installer\installer.bat" --run "%DEST%"

rem ---------------------------------------------------------------------------
:run
set "DEST=%~2"
set "WORK=%DEST%\_installer"
cd /d "%DEST%"
if exist "%DEST%\.git" goto :fail_git
if not exist "%DEST%\logs" mkdir "%DEST%\logs"
set "LOG=%DEST%\logs\installer.log"
echo [%DATE% %TIME%] installer started: %DEST% >> "%LOG%"
echo.

rem  Which version: the newest release on GitHub, or the newest code if there is no release yet.
set "TAG="
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (Invoke-RestMethod -UseBasicParsing -Headers @{ 'User-Agent' = 'PhotographicStudio-Installer' } -Uri ('https://api.github.com/repos/' + $env:REPO + '/releases/latest')).tag_name } catch { '' }"`) do set "TAG=%%T"
if defined TAG (
  set "ZIPURL=https://github.com/%REPO%/archive/refs/tags/%TAG%.zip"
  echo  1/4  Downloading Photographic Studio %TAG% from GitHub...
) else (
  set "ZIPURL=https://github.com/%REPO%/archive/refs/heads/%BRANCH%.zip"
  echo  1/4  Downloading the newest Photographic Studio from GitHub...
)
echo [%DATE% %TIME%] download %ZIPURL% >> "%LOG%"
call :download "%ZIPURL%" "%WORK%\studio.zip"
if errorlevel 1 goto :fail_net

echo  2/4  Unpacking...
if exist "%WORK%\src" rmdir /s /q "%WORK%\src"
mkdir "%WORK%\src"
call :unzip "%WORK%\studio.zip" "%WORK%\src"
set "SRC="
for /d %%D in ("%WORK%\src\*") do set "SRC=%%~fD"
if not defined SRC goto :fail_unzip
if not exist "%SRC%\studio\__init__.py" goto :fail_unzip

echo  3/4  Copying it into place (your settings, calibrations and photos are kept)...
rem  The download holds only software and shared profiles, so nothing of yours is
rem  overwritten, and nothing is deleted.
robocopy "%SRC%" "%DEST%" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >> "%LOG%" 2>&1
if errorlevel 8 goto :fail_copy
if defined TAG >"%DEST%\.installed_version" echo %TAG%
echo bat> "%DEST%\.install_kind"

echo  4/4  Setting up Python and the packages (the first time takes a few minutes)...
echo.
set "STUDIO_PRIVATE_PYTHON=1"
call "%DEST%\Install.bat" /nopause
if errorlevel 1 goto :fail_setup
rmdir /s /q "%WORK%\src" 2>nul
del /q "%WORK%\studio.zip" 2>nul
echo [%DATE% %TIME%] installer finished >> "%LOG%"

rem  Desktop shortcut: asked once (skipped when it is already there).
set "DESK="
for /f "usebackq delims=" %%K in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESK=%%K"
if not defined DESK goto :finished
if exist "%DESK%\Photographic Studio.lnk" goto :finished
echo.
choice /c YN /m "  Put a Photographic Studio shortcut on the desktop"
if errorlevel 2 goto :finished
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:DESK + '\Photographic Studio.lnk'); $s.TargetPath = $env:DEST + '\Start Studio.bat'; $s.WorkingDirectory = $env:DEST; $s.Description = 'Photographic Studio By Bill'; $s.Save()" >> "%LOG%" 2>&1

:finished
echo.
echo  =====================================================================
echo   All set. Photographic Studio is in:
echo      %DEST%
echo   Start it with the desktop shortcut or "Start Studio.bat" in that folder.
echo   To update later: turn on Help ^> Check for updates in the Studio, or run
echo   "Install Photographic Studio.bat" in that folder again.
echo  =====================================================================
echo.
choice /c YN /m "  Start the Studio now"
if errorlevel 2 goto :quit
start "" "%DEST%\Start Studio.bat"
endlocal
exit /b 0

:fail_git
echo.
echo  This folder is a git copy of the Studio; the installer does not touch it.
goto :quit
:fail_net
echo.
echo  PROBLEM: the download did not work. Check the internet connection and run
echo  the installer again. Details: %LOG%
goto :quit
:fail_unzip
echo.
echo  PROBLEM: the download could not be unpacked. Run the installer again.
echo  Details: %LOG%
goto :quit
:fail_copy
echo.
echo  PROBLEM: the files could not be copied into %DEST%.
echo  If the Studio is open, close it and run the installer again. Details: %LOG%
goto :quit
:fail_setup
echo.
echo  PROBLEM: setting up Python did not finish. Run the installer again; it
echo  carries on where it stopped. Details: logs\setup.log in %DEST%
goto :quit

:quit
echo.
pause
endlocal
exit /b 1

rem ---------------------------------------------------------------------------
rem  :download URL FILE - curl.exe (in Windows 10 and 11), else PowerShell.
:download
set "DL_URL=%~1"
set "DL_OUT=%~2"
if exist "%DL_OUT%" del /q "%DL_OUT%"
where curl.exe >nul 2>nul
if errorlevel 1 goto :download_ps
curl.exe -L --fail --retry 3 --retry-delay 2 -# -o "%DL_OUT%" "%DL_URL%"
if not errorlevel 1 if exist "%DL_OUT%" exit /b 0
:download_ps
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -UseBasicParsing -Uri $env:DL_URL -OutFile $env:DL_OUT; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 exit /b 1
if not exist "%DL_OUT%" exit /b 1
exit /b 0

rem  :unzip ZIP FOLDER - tar.exe (in Windows 10 and 11), else PowerShell.
:unzip
set "UZ_ZIP=%~1"
set "UZ_DIR=%~2"
tar -xf "%UZ_ZIP%" -C "%UZ_DIR%" >> "%LOG%" 2>&1
if not errorlevel 1 exit /b 0
powershell -NoProfile -Command "Expand-Archive -Force -LiteralPath $env:UZ_ZIP -DestinationPath $env:UZ_DIR" >> "%LOG%" 2>&1
exit /b %errorlevel%
