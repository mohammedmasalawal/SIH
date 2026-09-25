"""FastAPI entry point. Business logic lives under backend/ -- keep this file thin."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import routes
from backend.config import API_HOST, API_PORT, CORS_ORIGINS, FRONTEND_DIR, SITE_DIST_DIR

app = FastAPI(title="Agninetra Thermal Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    """Frontend files carry no Cache-Control by default, so browsers reuse stale JS/CSS
    after an update until a hard refresh; no-cache makes them revalidate each load
    (cheap: unchanged files answer 304 via their ETag)."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


# The dashboard reads only static files under data/ -- the same export (python -m
# training.export_site) that is deployed to Vercel -- so local and hosted behave alike.
app.mount("/data", StaticFiles(directory=SITE_DIST_DIR / "data", check_dir=False), name="site-data")
# Served same-origin so the legacy page's fetch("/api/...") calls need no CORS config.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
