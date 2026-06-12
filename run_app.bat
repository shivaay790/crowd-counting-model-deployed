@echo off
echo Starting Crowd Counting Project...

:: Start Backend with Venv
start cmd /k "cd backend && venv\Scripts\activate && python app.py"

:: Start Frontend
start cmd /k "cd frontend && npm run dev"

echo Backend and Frontend are starting in separate windows.
echo Backend: http://localhost:4000
echo Frontend: http://localhost:4001
pause
