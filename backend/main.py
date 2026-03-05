from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from backend.db.init_db import init_db
from backend.api.routes.candles import router as candles_router
from backend.api.routes.ingest import router as ingest_router
from backend.api.routes.features import router as features_router
from backend.api.routes.signals_simple import router as signals_router
from backend.api.routes.scan import router as scan_router
from backend.api.routes.build import router as build_router
from backend.api.routes.performance import router as performance_router
from backend.api.routes.setups import router as setups_router

app = FastAPI(title="Trading App API")

# register routers
app.include_router(candles_router)
app.include_router(ingest_router)
app.include_router(features_router)
app.include_router(signals_router)
app.include_router(scan_router)
app.include_router(build_router)
app.include_router(performance_router)
app.include_router(setups_router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}