@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found in PATH. Please install Python and try again.
    pause
    exit /b 1
)

set DEFAULT_DATA=%~dp0..\temp_train_data
if "%~1"=="" (
    python viewer.py "%DEFAULT_DATA%"
) else (
    python viewer.py %*
)
if errorlevel 1 (
    echo Failed to launch viewer. Check that PySide6, numpy, astropy, and matplotlib are installed.
    pause
)
