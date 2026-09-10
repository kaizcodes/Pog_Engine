@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Pog Engine Updater

echo ============================================================
echo  Pog Engine - Differential Updater
echo ============================================================
echo.
echo This checks the latest public release on GitHub and replaces
echo ONLY the code files that changed. Your models, VOD folders,
echo config tunings, and machine paths are left alone.
echo.

rem ---- Locate Python, or install it if nothing is found ---------------------
set "PY_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 --version >nul 2>nul
        if not errorlevel 1 (
            for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PY_CMD=%%P"
        )
    )
)

if not defined PY_CMD (
    echo Python was not found on this PC - downloading and installing Python 3.12.10 ...
    call :INSTALL_PY312
    if errorlevel 1 exit /b 1
    set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "!PY_CMD!" (
        set "PY_CMD="
        where python >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
    if not defined PY_CMD (
        echo.
        echo ERROR: Python was installed but could not be located automatically.
        echo Close this window, open a NEW Command Prompt, and run this updater again.
        pause
        exit /b 1
    )
)
rem ---- Require Python 3.12 - older interpreters are no longer used ------------
set "PY_VER="
for /f "tokens=2" %%V in ('"%PY_CMD%" --version 2^>^&1') do set "PY_VER=%%V"
set "PY_MAJOR="
set "PY_MINOR="
for /f "tokens=1 delims=." %%A in ("%PY_VER%") do set "PY_MAJOR=%%A"
for /f "tokens=2 delims=." %%A in ("%PY_VER%") do set "PY_MINOR=%%A"
if not defined PY_MINOR set "PY_MINOR=0"
set "NEED_PY312="
if not "%PY_MAJOR%"=="3" set "NEED_PY312=1"
if "%PY_MAJOR%"=="3" if not "%PY_MINOR%"=="12" set "NEED_PY312=1"
if defined NEED_PY312 (
    echo.
    echo Found Python %PY_VER% - Pog Engine now requires Python 3.12+.
    echo Downloading and installing Python 3.12.10 - your existing Python is left untouched ...
    call :INSTALL_PY312
    if errorlevel 1 exit /b 1
    set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "!PY_CMD!" (
        echo.
        echo ERROR: Python 3.12 was installed but could not be located automatically.
        echo Close this window, open a NEW Command Prompt, and run this updater again.
        pause
        exit /b 1
    )
    echo.
)

echo Using Python:
"%PY_CMD%" --version
echo.

rem ---- Hand off to the updater (extra words after the .bat pass through) -----
rem ---- e.g. Update_PogEngine.bat --check-only --------------------------------
"%PY_CMD%" "%~dp0pog_engine_update.py" %*
set "UPDATE_RESULT=%ERRORLEVEL%"

echo.
echo ============================================================
if "%UPDATE_RESULT%"=="0" (
    echo  Updater finished. Re-run any time to check for new releases.
) else (
    echo  The updater exited with an error - see above for details.
)
echo ============================================================
pause
exit /b %UPDATE_RESULT%

rem ---- Download + silently install Python 3.12.10 (shared by both paths above)
:INSTALL_PY312
set "PY_INSTALLER=%TEMP%\python-3.12.10-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '!PY_INSTALLER!'"
if not exist "!PY_INSTALLER!" (
    echo.
    echo ERROR: could not download the Python installer.
    echo Check your internet connection, or install Python 3.12 yourself
    echo from https://www.python.org/downloads/ and re-run this script.
    pause
    exit /b 1
)
echo Installing Python silently - this can take a minute or two...
"!PY_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Include_test=0
exit /b 0
