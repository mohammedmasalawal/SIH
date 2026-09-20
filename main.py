"""FastAPI entry point. Business logic lives under backend/ -- keep this file thin."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import routes
from backend.config import API_HOST, API_PORT, CORS_ORIGINS, FRONTEND_DIR

app = FastAPI(title="Agninetra Thermal Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)
# Served same-origin so the frontend's fetch("/api/...") calls need no CORS config.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
