@echo off
rem Live ingestion entry point for Windows Task Scheduler (see README "Live ingestion").
rem Runs from the repo root so python -m and the .env FIRMS_MAP_KEY lookup resolve.
rem Pulls the last 3 days every run (runs are 6 h apart): overlapping windows are free
rem -- already-stored detections are skipped -- and a missed run or a day with the
rem machine off doesn't leave a gap. Extra arguments are passed through (a later --days wins).
cd /d "%~dp0.."
if not exist "training\data\live\logs" mkdir "training\data\live\logs"
echo ===== %date% %time% >> "training\data\live\logs\ingest.log"
rem Refuse to run on a missing or half-restored data folder: the OSM extract, its context parquet and the
rem WorldCover tiles must exist and match data\MANIFEST.json (sizes). On a mismatch: a clear message in the
rem log, a heartbeat failure ping if HEARTBEAT_URL is set, and exit code 2 -- no ingest, no export, no deploy.
".venv\Scripts\python.exe" -m training.data_manifest check >> "training\data\live\logs\ingest.log" 2>&1
if errorlevel 1 (
  echo INGEST ABORTED: input check failed, see the lines above >> "training\data\live\logs\ingest.log"
  exit /b 2
)
".venv\Scripts\python.exe" -m training.ingest_latest --days 3 %* >> "training\data\live\logs\ingest.log" 2>&1
set INGEST_EXIT=%errorlevel%
rem Rebuild the static dashboard and deploy it to Vercel -- only when data, alerts or the
rem page changed. Authenticates with VERCEL_TOKEN from .env (not the interactive login);
rem failed deploys are retried, then logged with the CLI's output, and never fail the run.
".venv\Scripts\python.exe" -m training.export_site --deploy --if-changed --require-token >> "training\data\live\logs\ingest.log" 2>&1
exit /b %INGEST_EXIT%
