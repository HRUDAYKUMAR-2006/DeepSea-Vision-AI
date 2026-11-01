@echo off
title DeepSea Vision AI - Marine Search & Rescue Platform
echo ===============================================================================
echo     DEEPSEA VISION AI - MARINE SEARCH PARTY & UNDERWATER SURVEY SYSTEM
echo ===============================================================================
echo Starting local web engine and mission dashboard...
cd /d "%~dp0"

REM Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ from python.org to run this platform.
    pause
    exit /b
)

echo Initializing database storage and starting application server...
start "" "http://127.0.0.1:5000"
python demo/app.py
pause
