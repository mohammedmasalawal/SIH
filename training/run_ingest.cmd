@echo off
rem Live ingestion entry point for Windows Task Scheduler (see README "Live ingestion").
rem Runs from the repo root so python -m and the .env FIRMS_MAP_KEY lookup resolve.
cd /d "%~dp0.."
if not exist "training\data\live\logs" mkdir "training\data\live\logs"
echo ===== %date% %time% >> "training\data\live\logs\ingest.log"
".venv\Scripts\python.exe" -m training.ingest_latest %* >> "training\data\live\logs\ingest.log" 2>&1
exit /b %errorlevel%
