@echo off
echo Starting GolfAI...
echo.

:: Start backend
start "GolfAI Backend" cmd /k "cd /d %~dp0backend && pip install -r requirements.txt -q && python main.py"

:: Wait a moment then start frontend
timeout /t 3 /nobreak > nul
start "GolfAI Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Both servers are starting in separate windows.
