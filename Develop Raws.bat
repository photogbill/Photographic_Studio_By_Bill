@echo off
setlocal
title Photographic Studio By Bill - Develop Raws
rem ---------------------------------------------------------------------------
rem  Develop Raws.bat
rem    Drag CR2 / CR3 files (or a folder) onto this file: each one becomes a
rem    full-size 16-bit TIFF in Adobe RGB, and a comparison page opens showing
rem    the Studio result next to the camera's own JPEG.
rem    Double-click it to try the sample raws in this folder.
rem  Results go to outputs\develop in this folder (never to C:).
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
call "%~dp0tools\ensure_env.bat"
if errorlevel 1 goto :fail

if not "%~1"=="" goto :dropped

echo.
echo  Photographic Studio By Bill - develop raws
echo  ==========================================
echo  Tip: you can drag raw files or a whole folder onto this .bat file.
echo.
echo  No files were dropped. What would you like to do with the raws in this folder?
echo    C = comparison page only (quick: Studio vs camera JPEG, no big files)
echo    T = full-size 16-bit TIFFs AND the comparison page (about 250 MB per photo)
echo    Q = quit
echo.
choice /c CTQ /n /m "  Press C, T or Q: "
if errorlevel 3 goto :end
if errorlevel 2 goto :samples_tiff
"%PY%" -m studio develop "%~dp0." --compare-only
if errorlevel 1 goto :fail
goto :done

:samples_tiff
"%PY%" -m studio develop "%~dp0." --compare
if errorlevel 1 goto :fail
goto :done

:dropped
"%PY%" -m studio develop %* --compare
if errorlevel 1 goto :fail

:done
echo.
echo  Finished. The files are in the outputs\develop folder.
echo.
pause
goto :end

:fail
echo.
echo  Something went wrong. Details are in logs\studio.log
echo  (Help: run "%~dp0.venv\Scripts\python.exe" -m studio diag  for a report you can send.)
echo.
pause
endlocal
exit /b 1

:end
endlocal
exit /b 0
