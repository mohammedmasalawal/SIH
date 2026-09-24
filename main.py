"""FastAPI entry point. Business logic lives under backend/ -- keep this file thin."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import routes
from backend.config import ALERTS_HISTORY_PATH, API_HOST, API_PORT, CORS_ORIGINS, FRONTEND_DIR

app = FastAPI(title="Agninetra Thermal Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)


@app.get("/data/alerts_history.csv", include_in_schema=False)
def alerts_history() -> FileResponse:
    """The alerts panel's data, at the path a static host serves it from
    (data/alerts_history.csv beside index.html); locally it's the ingest output."""
    if not ALERTS_HISTORY_PATH.exists():
        raise HTTPException(status_code=404, detail="no alerts recorded yet")
    return FileResponse(ALERTS_HISTORY_PATH, media_type="text/csv", headers={"Cache-Control": "no-cache"})


# Served same-origin so the frontend's fetch("/api/...") calls need no CORS config.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
