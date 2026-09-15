"""FastAPI entry point. Business logic lives under backend/ -- keep this file thin."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes
from backend.config import API_HOST, API_PORT, CORS_ORIGINS

app = FastAPI(title="Agninetra Thermal Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
