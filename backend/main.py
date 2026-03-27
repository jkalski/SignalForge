from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from backend.api.routes.probability import router as probability_router
from backend.api.routes.analyze import router as analyze_router
from backend.api.routes.notifications import router as notifications_router
from backend.scheduler.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Trading App API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candles_router)
app.include_router(ingest_router)
app.include_router(features_router)
app.include_router(signals_router)
app.include_router(scan_router)
app.include_router(build_router)
app.include_router(performance_router)
app.include_router(setups_router)
app.include_router(probability_router)
app.include_router(analyze_router)
app.include_router(notifications_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}