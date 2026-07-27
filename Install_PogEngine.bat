@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Pog_Engine Installer

echo ============================================================
echo  Pog_Engine - Setup / Dependency Installer
echo ============================================================
echo.
echo Launching the installer window - use it to confirm your
echo Pog_Engine folder and watch progress for each check.
echo (This console window stays open behind it in case anything
echo needs to be visible here too, e.g. if the window fails to open.)
echo.

set "DEFAULT_DIR=%~dp0"
set "DEFAULT_DIR=%DEFAULT_DIR:~0,-1%"

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
    echo Python was not found on this PC - downloading and installing Python 3.11.9 ...
    set "PY_INSTALLER=%TEMP%\python-3.11.9-amd64.exe"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '!PY_INSTALLER!'"

    if not exist "!PY_INSTALLER!" (
        echo.
        echo ERROR: could not download the Python installer.
        echo Check your internet connection, or install Python 3.10+ yourself
        echo from https://www.python.org/downloads/ and re-run this script.
        pause
        exit /b 1
    )

    echo Installing Python silently - this can take a minute or two...
    "!PY_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Include_test=0

    set "PY_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
    if not exist "!PY_CMD!" (
        set "PY_CMD="
        where python >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
    if not defined PY_CMD (
        echo.
        echo ERROR: Python was installed but could not be located automatically.
        echo Close this window, open a NEW Command Prompt, and run this installer again.
        pause
        exit /b 1
    )
)

echo Using Python:
"%PY_CMD%" --version
echo.

rem ---- Hand off to the setup / verification script ---------------------------
"%PY_CMD%" "%~dp0pog_engine_setup.py" "%DEFAULT_DIR%"
set "SETUP_RESULT=%ERRORLEVEL%"

echo.
echo ============================================================
if "%SETUP_RESULT%"=="0" (
    echo  Installer window closed. Re-run this .bat any time to re-check.
) else (
    echo  The installer exited with an error - see above for details.
)
echo ============================================================
pause
exit /b %SETUP_RESULT%
