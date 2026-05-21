"""
FastAPI application — MovieRec API.

Wires together all routers, middleware, and the model-loading lifespan.
"""

import os
import sys

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Load environment variables before anything else
load_dotenv()

# Ensure project root is on sys.path for clean imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts into memory on startup."""
    from api.model_registry import ModelRegistry

    logger.info("🚀 Starting MovieRec API — loading model artifacts…")

    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry

    logger.info(
        f"✅ Startup complete — "
        f"TwoTower={'✓' if registry.two_tower else '✗'}  "
        f"Ranker={'✓' if registry.ranker else '✗'}  "
        f"FAISS={'✓' if registry.faiss_index else '✗'}  "
        f"FeatureStore={'✓' if registry.feature_store else '✗'}"
    )

    yield  # application runs here

    logger.info("👋 Shutting down MovieRec API")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MovieRec API",
    description="Multi-stage recommender system with Two-Tower retrieval, LightGBM ranking, and MMR re-ranking.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit frontend (and any other origin for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from api.routers import feedback, health, recommend, similar  # noqa: E402

app.include_router(recommend.router, prefix="/api/v1")
app.include_router(similar.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(health.router)  # health lives at /health (no prefix)
