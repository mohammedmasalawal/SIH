@echo off
rem Live ingestion entry point for Windows Task Scheduler (see README "Live ingestion").
rem Runs from the repo root so python -m and the .env FIRMS_MAP_KEY lookup resolve.
rem Pulls the last 3 days every run (runs are 6 h apart): overlapping windows are free
rem -- already-stored detections are skipped -- and a missed run or a day with the
rem machine off doesn't leave a gap. Extra arguments are passed through (a later --days wins).
cd /d "%~dp0.."
if not exist "training\data\live\logs" mkdir "training\data\live\logs"
echo ===== %date% %time% >> "training\data\live\logs\ingest.log"
".venv\Scripts\python.exe" -m training.ingest_latest --days 3 %* >> "training\data\live\logs\ingest.log" 2>&1
set INGEST_EXIT=%errorlevel%
rem Rebuild the static dashboard and deploy it to Vercel -- only when data, alerts or the
rem page changed. Authenticates with VERCEL_TOKEN from .env (not the interactive login);
rem failed deploys are retried, then logged with the CLI's output, and never fail the run.
".venv\Scripts\python.exe" -m training.export_site --deploy --if-changed --require-token >> "training\data\live\logs\ingest.log" 2>&1
exit /b %INGEST_EXIT%
