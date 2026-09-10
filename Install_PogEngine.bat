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

rem ---- Locate Python 3.12 - reuse an existing install, download only if none -
set "PY_CMD="
set "FOUND_VER="
set "PY_MAJOR="
set "PY_MINOR="

rem 1) python on PATH, but only when it is already 3.12
where python >nul 2>nul
if not errorlevel 1 call :TRY_EXE python

rem 2) exact-version py launcher, unaffected by which Python is newest
if not defined PY_CMD (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PY_CMD=%%P"
    )
)

rem 3) well-known install paths, for a 3.12 that is neither on PATH nor the default py
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python312\python.exe" call :TRY_EXE "%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PY_CMD if exist "C:\Python312\python.exe" call :TRY_EXE "C:\Python312\python.exe"
if not defined PY_CMD if exist "C:\Program Files\Python312\python.exe" call :TRY_EXE "C:\Program Files\Python312\python.exe"

rem 4) no 3.12 anywhere - report what was found, then download 3.12.10
if not defined PY_CMD (
    if not defined FOUND_VER (
        where python >nul 2>nul
        if not errorlevel 1 (
            for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "FOUND_VER=%%V"
        )
    )
    if not defined FOUND_VER (
        where py >nul 2>nul
        if not errorlevel 1 (
            for /f "tokens=2" %%V in ('py -3 --version 2^>^&1') do set "FOUND_VER=%%V"
        )
    )
    if defined FOUND_VER call :PARSE_VER "!FOUND_VER!"
    echo.
    if defined FOUND_VER (
        echo Found Python !FOUND_VER! - Pog Engine now requires Python 3.12 ...
    ) else (
        echo Python 3.12 was not found on this PC - downloading Python 3.12.10 ...
    )
    echo Downloading and installing Python 3.12.10 - your existing Python is left untouched ...
    call :INSTALL_PY312
    if errorlevel 1 exit /b 1
    set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "!PY_CMD!" (
        echo.
        echo ERROR: Python 3.12 was installed but could not be located automatically.
        echo Close this window, open a NEW Command Prompt, and run this installer again.
        pause
        exit /b 1
    )
    echo.
    if defined FOUND_VER (
        echo RECOMMENDED: uninstall the old Python !PY_MAJOR!.!PY_MINOR! to avoid mixing interpreters.
        echo   Settings ^> Apps ^> Installed apps ^> Python !PY_MAJOR!.!PY_MINOR! ^> Uninstall
    )
    choice /T 15 /D N /M "Open Installed apps for you now - auto-No in 15s"
    if not errorlevel 2 start ms-settings:appsfeatures
    echo.
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
rem ---- Adopt one Python exe when it is 3.12, else remember its version for the message
:TRY_EXE
set "TRY_VER="
for /f "tokens=2" %%V in ('"%~1" --version 2^>^&1') do set "TRY_VER=%%V"
if not defined TRY_VER exit /b 1
call :PARSE_VER "!TRY_VER!"
if "!PY_MAJOR!"=="3" if "!PY_MINOR!"=="12" (
    set "PY_CMD=%~1"
    set "FOUND_VER=!TRY_VER!"
    exit /b 0
)
if not defined FOUND_VER set "FOUND_VER=!TRY_VER!"
exit /b 1

rem ---- Split a dotted version into PY_MAJOR / PY_MINOR
:PARSE_VER
set "PY_MAJOR="
set "PY_MINOR="
for /f "tokens=1 delims=." %%A in ("%~1") do set "PY_MAJOR=%%A"
for /f "tokens=2 delims=." %%A in ("%~1") do set "PY_MINOR=%%A"
if not defined PY_MINOR set "PY_MINOR=0"
exit /b 0
