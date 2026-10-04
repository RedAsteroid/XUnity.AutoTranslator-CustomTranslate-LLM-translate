@echo off
cd /d "%~dp0"

echo ============================================
echo   Starting translation service ...
echo ============================================

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

python main.py

echo.
echo Service stopped. Please read the log above.
pause