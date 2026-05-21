"""
GET /api/v1/similar/{movie_id}

Returns the top-N most similar movies to a given movie using
cosine similarity on item embeddings stored in the FAISS index.
Results are cached for 24 hours.
"""

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel

from api.cache import TTL_SIMILAR, get_cached, set_cached

router = APIRouter(tags=["similar"])

# ---------------------------------------------------------------------------
# Canonical genre list (must match the rest of the project)
# ---------------------------------------------------------------------------
GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimilarMovieItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: list[str]
    score: float


class SimilarMoviesResponse(BaseModel):
    movie_id: int
    title: str
    similar_movies: list[SimilarMovieItem]
    latency_ms: float
    cache_hit: bool


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
async def similar_movies(
    request: Request,
    movie_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Number of similar movies"),
) -> SimilarMoviesResponse:
    t0 = time.perf_counter()
    registry = request.app.state.registry

    # ── Validate movie exists ─────────────────────────────────────────
    if registry.feature_store is None or movie_id not in registry.feature_store.item_features:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in the system.")

    if registry.faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded. Check /health.")

    # ── Cache check ───────────────────────────────────────────────────
    cache_key = f"sim:{movie_id}:{top_n}"
    cached = get_cached(cache_key)
    if cached is not None:
        latency = (time.perf_counter() - t0) * 1000
        query_feats = registry.feature_store.item_features.get(movie_id, {})
        query_title = query_feats.get("title", f"Movie {movie_id}")
        return SimilarMoviesResponse(
            movie_id=movie_id,
            title=query_title,
            similar_movies=[SimilarMovieItem(**m) for m in cached],
            latency_ms=round(latency, 1),
            cache_hit=True,
        )

    # ── Look up the query movie's embedding in the FAISS index ────────
    try:
        faiss_idx = registry.faiss_index.item_ids.index(movie_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} has no embedding in the FAISS index.",
        )

    # Reconstruct the query embedding from the FAISS index
    query_emb = registry.faiss_index.index.reconstruct(faiss_idx)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)

    # Search for top_n + 1 (the query movie itself will appear as the #1 result)
    scores_arr, indices_arr = registry.faiss_index.index.search(query_emb, top_n + 1)

    # ── Build result list (skip the query movie itself) ───────────────
    query_feats = registry.feature_store.item_features.get(movie_id, {})
    query_title = query_feats.get("title", f"Movie {movie_id}")

    similar: list[dict] = []
    rank = 1
    for idx, score in zip(indices_arr[0], scores_arr[0]):
        if idx < 0 or idx >= len(registry.faiss_index.item_ids):
            continue
        mid = registry.faiss_index.item_ids[idx]
        if mid == movie_id:
            continue  # skip self
        if rank > top_n:
            break

        # Look up metadata
        if registry.item_meta is not None and mid in registry.item_meta.index:
            meta = registry.item_meta.loc[mid]
            title = meta.get("title", f"Movie {mid}")
            genres = meta.get("genres", [])
        else:
            item_feats = registry.feature_store.item_features.get(mid, {})
            title = item_feats.get("title", f"Movie {mid}")
            genre_vec = item_feats.get("genre_vec", np.zeros(18))
            genres = [GENRES[i] for i, v in enumerate(genre_vec) if v > 0]

        similar.append({
            "rank": rank,
            "movie_id": mid,
            "title": title,
            "genres": list(genres) if not isinstance(genres, list) else genres,
            "score": round(float(score), 4),
        })
        rank += 1

    # ── Cache ─────────────────────────────────────────────────────────
    set_cached(cache_key, similar, TTL_SIMILAR)

    latency = (time.perf_counter() - t0) * 1000
    return SimilarMoviesResponse(
        movie_id=movie_id,
        title=query_title,
        similar_movies=[SimilarMovieItem(**m) for m in similar],
        latency_ms=round(latency, 1),
        cache_hit=False,
    )
