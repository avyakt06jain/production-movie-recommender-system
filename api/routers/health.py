"""
GET /health

Health-check endpoint (no /api/v1 prefix).
Returns model load status and FAISS index size.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    faiss_index_size: int


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "registry", None)

    if registry is None:
        return HealthResponse(status="ok", models_loaded=False, faiss_index_size=0)

    models_loaded = (
        registry.two_tower is not None
        and registry.ranker is not None
        and registry.faiss_index is not None
        and registry.feature_store is not None
    )

    faiss_size = 0
    if registry.faiss_index is not None:
        faiss_size = registry.faiss_index.index.ntotal

    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        faiss_index_size=faiss_size,
    )
