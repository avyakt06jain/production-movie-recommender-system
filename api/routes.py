"""
API Routes — all endpoints for the MovieRec API.

Endpoints:
  GET  /health                        — Health check
  GET  /api/v1/recommend/{user_id}    — Personalized recommendations
  GET  /api/v1/similar/{movie_id}     — Similar movies
  POST /api/v1/feedback               — User feedback ingestion
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache import (
    TTL_RECS,
    TTL_SIMILAR,
    get_cached,
    invalidate_user_cache,
    set_cached,
)
from api.database import get_db

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
api_router = APIRouter(prefix="/api/v1")
health_router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    faiss_index_size: int


@health_router.get("/health", response_model=HealthResponse, tags=["health"])
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


# ═══════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════

class RecommendationItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: list[str]
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    latency_ms: float
    cache_hit: bool


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


def _mmr_rerank(
    candidates: list[dict],
    top_n: int = 10,
    lambda_mmr: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking for diversity."""
    if not candidates:
        return []

    selected: list[dict] = []
    remaining = candidates.copy()

    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x["score"])
        else:
            def mmr_score(item: dict) -> float:
                rel = item["score"]
                max_sim = max(
                    _cosine_similarity(item["genre_vec"], s["genre_vec"])
                    for s in selected
                )
                return lambda_mmr * rel - (1 - lambda_mmr) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


def _build_user_tensor(user_feats: dict, registry) -> dict:
    """Build the dict of tensors expected by UserTower.forward()."""
    return {
        "user_id": torch.tensor([user_feats.get("user_id", 0)], dtype=torch.long),
        "age_bucket": torch.tensor([user_feats.get("age_bucket", 0)], dtype=torch.long),
        "gender": torch.tensor([user_feats.get("gender", 0)], dtype=torch.long),
        "occupation": torch.tensor([user_feats.get("occupation", 0)], dtype=torch.long),
        "watched_genre_vec": torch.tensor(
            [user_feats.get("genre_pref_vec", np.zeros(18))], dtype=torch.float32
        ),
    }


def _build_rank_features(
    user_feats: dict,
    candidate_ids: list[int],
    candidate_scores: dict[int, float],
    registry,
) -> np.ndarray:
    """Build the (N, F) feature matrix for LightGBM ranking."""
    user_genre_pref = np.array(user_feats.get("genre_pref_vec", np.zeros(18)), dtype=np.float32)
    user_static = np.array([
        user_feats.get("age_norm", 0.0),
        user_feats.get("gender_enc", 0),
        user_feats.get("occupation", 0),
        user_feats.get("avg_rating", 3.0),
        user_feats.get("rating_count", 0),
    ], dtype=np.float32)

    feature_rows: list[np.ndarray] = []
    for mid in candidate_ids:
        item_feats = registry.feature_store.item_features.get(mid, {})
        item_genre_vec = np.array(item_feats.get("genre_vec", np.zeros(18)), dtype=np.float32)

        item_static = np.array([
            item_feats.get("avg_rating", 0.0),
            item_feats.get("rating_count_log", 0.0),
            item_feats.get("year_norm", 0.0),
        ], dtype=np.float32)

        two_tower_score = candidate_scores.get(mid, 0.0)
        genre_overlap = float(np.dot(user_genre_pref, item_genre_vec))
        user_has_rated_genre = 0.0
        popularity_pct = item_feats.get("popularity_pct", 0.5)
        is_sequel = 0.0

        row = np.concatenate([
            user_static,           # 5
            user_genre_pref,       # 18
            item_static,           # 3
            item_genre_vec,        # 18
            np.array([two_tower_score, genre_overlap, user_has_rated_genre, popularity_pct, is_sequel], dtype=np.float32),  # 5
        ])
        feature_rows.append(row)

    if not feature_rows:
        return np.empty((0, 49), dtype=np.float32)

    return np.vstack(feature_rows).astype(np.float32)


@api_router.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
async def recommend(
    request: Request,
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    exclude_watched: bool = Query(default=True, description="Filter already-seen movies"),
    diversity: float = Query(
        default=0.7, ge=0.0, le=1.0,
        description="MMR lambda: 0 = pure diversity, 1 = pure relevance",
    ),
) -> RecommendationResponse:
    t0 = time.perf_counter()
    registry = request.app.state.registry

    # Validate user exists
    if registry.feature_store is None or user_id not in registry.feature_store.user_features:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in the system.")

    # Stage 0: Cache check
    cache_key = f"rec:{user_id}:{top_n}"
    cached = get_cached(cache_key)
    if cached is not None:
        latency = (time.perf_counter() - t0) * 1000
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[RecommendationItem(**r) for r in cached],
            latency_ms=round(latency, 1),
            cache_hit=True,
        )

    # Stage 1: Candidate generation (Two-Tower → FAISS)
    user_feats = registry.feature_store.user_features[user_id]

    if registry.two_tower is None or registry.faiss_index is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health for status.",
        )

    faiss_top_k = int(os.environ.get("FAISS_TOP_K", 200))
    user_batch = _build_user_tensor(user_feats, registry)

    with torch.no_grad():
        user_emb = registry.two_tower.user_tower(**user_batch)

    user_emb_np = user_emb.numpy().astype(np.float32)

    scores_arr, indices_arr = registry.faiss_index.index.search(
        user_emb_np.reshape(1, -1), faiss_top_k
    )
    candidate_ids: list[int] = []
    candidate_scores: dict[int, float] = {}
    for idx, score in zip(indices_arr[0], scores_arr[0]):
        if 0 <= idx < len(registry.faiss_index.item_ids):
            mid = registry.faiss_index.item_ids[idx]
            candidate_ids.append(mid)
            candidate_scores[mid] = float(score)

    # Stage 2: Ranking (LightGBM)
    ranker_top_k = int(os.environ.get("RANKER_TOP_K", 20))

    if registry.ranker is not None and candidate_ids:
        rank_features = _build_rank_features(
            user_feats, candidate_ids, candidate_scores, registry
        )
        lgbm_scores = registry.ranker.predict(rank_features)

        scored_pairs = sorted(
            zip(candidate_ids, lgbm_scores), key=lambda x: x[1], reverse=True
        )
        candidate_ids = [mid for mid, _ in scored_pairs[:ranker_top_k]]
        candidate_scores = {mid: float(sc) for mid, sc in scored_pairs[:ranker_top_k]}
    else:
        scored_pairs = sorted(
            ((mid, candidate_scores.get(mid, 0.0)) for mid in candidate_ids),
            key=lambda x: x[1],
            reverse=True,
        )
        candidate_ids = [mid for mid, _ in scored_pairs[:ranker_top_k]]
        candidate_scores = {mid: sc for mid, sc in scored_pairs[:ranker_top_k]}

    # Stage 3: Re-ranking (watched filter + MMR)
    watched_set: set[int] = set()
    if exclude_watched:
        watched_set = set(user_feats.get("rated_movie_ids", []))

    if candidate_scores:
        max_score = max(candidate_scores.values())
        min_score = min(candidate_scores.values())
        score_range = max_score - min_score if max_score != min_score else 1.0
    else:
        max_score = min_score = score_range = 1.0

    rerank_candidates: list[dict] = []
    for mid in candidate_ids:
        if mid in watched_set:
            continue

        item_feats = registry.feature_store.item_features.get(mid, {})
        genre_vec = np.array(item_feats.get("genre_vec", np.zeros(18)), dtype=np.float32)
        norm_score = (candidate_scores.get(mid, 0.0) - min_score) / score_range

        rerank_candidates.append({
            "movie_id": mid,
            "score": norm_score,
            "genre_vec": genre_vec,
        })

    final_items = _mmr_rerank(rerank_candidates, top_n=top_n, lambda_mmr=diversity)

    # Build response
    recommendations: list[dict] = []
    for rank_idx, item in enumerate(final_items, 1):
        mid = item["movie_id"]
        if registry.item_meta is not None and mid in registry.item_meta.index:
            meta = registry.item_meta.loc[mid]
            title = meta.get("title", f"Movie {mid}")
            genres = meta.get("genres", [])
        else:
            item_feats = registry.feature_store.item_features.get(mid, {})
            title = item_feats.get("title", f"Movie {mid}")
            genre_vec = item_feats.get("genre_vec", np.zeros(18))
            genres = [GENRES[i] for i, v in enumerate(genre_vec) if v > 0]

        rec = {
            "rank": rank_idx,
            "movie_id": mid,
            "title": title,
            "genres": list(genres) if not isinstance(genres, list) else genres,
            "score": round(item["score"], 4),
        }
        recommendations.append(rec)

    set_cached(cache_key, recommendations, TTL_RECS)

    latency = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[RecommendationItem(**r) for r in recommendations],
        latency_ms=round(latency, 1),
        cache_hit=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Similar Movies
# ═══════════════════════════════════════════════════════════════════════════

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


@api_router.get("/similar/{movie_id}", response_model=SimilarMoviesResponse, tags=["similar"])
async def similar_movies(
    request: Request,
    movie_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Number of similar movies"),
) -> SimilarMoviesResponse:
    t0 = time.perf_counter()
    registry = request.app.state.registry

    if registry.feature_store is None or movie_id not in registry.feature_store.item_features:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in the system.")

    if registry.faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded. Check /health.")

    # Cache check
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

    # FAISS lookup
    try:
        faiss_idx = registry.faiss_index.item_ids.index(movie_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} has no embedding in the FAISS index.",
        )

    query_emb = registry.faiss_index.index.reconstruct(faiss_idx)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)

    scores_arr, indices_arr = registry.faiss_index.index.search(query_emb, top_n + 1)

    query_feats = registry.feature_store.item_features.get(movie_id, {})
    query_title = query_feats.get("title", f"Movie {movie_id}")

    similar: list[dict] = []
    rank = 1
    for idx, score in zip(indices_arr[0], scores_arr[0]):
        if idx < 0 or idx >= len(registry.faiss_index.item_ids):
            continue
        mid = registry.faiss_index.item_ids[idx]
        if mid == movie_id:
            continue
        if rank > top_n:
            break

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

    set_cached(cache_key, similar, TTL_SIMILAR)

    latency = (time.perf_counter() - t0) * 1000
    return SimilarMoviesResponse(
        movie_id=movie_id,
        title=query_title,
        similar_movies=[SimilarMovieItem(**m) for m in similar],
        latency_ms=round(latency, 1),
        cache_hit=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════════════

VALID_EVENT_TYPES = {"click", "watch", "skip", "rate"}


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user")
    movie_id: int = Field(..., description="ID of the movie")
    event_type: str = Field(
        ...,
        description="Type of interaction: click, watch, skip, or rate",
    )
    session_id: str = Field(..., description="Client session identifier")
    rating: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="Rating value (required for 'rate' events)"
    )


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    event_id: Optional[int] = None


@api_router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def submit_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    if body.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid event_type '{body.event_type}'. Must be one of: {sorted(VALID_EVENT_TYPES)}",
        )

    if body.event_type == "rate" and body.rating is None:
        raise HTTPException(
            status_code=422,
            detail="A 'rating' value (1.0-5.0) is required for 'rate' events.",
        )

    insert_sql = text("""
        INSERT INTO feedback_events (user_id, movie_id, event_type, rating, session_id, timestamp)
        VALUES (:user_id, :movie_id, :event_type, :rating, :session_id, :ts)
        RETURNING event_id
    """)

    try:
        result = await db.execute(
            insert_sql,
            {
                "user_id": body.user_id,
                "movie_id": body.movie_id,
                "event_type": body.event_type,
                "rating": body.rating,
                "session_id": body.session_id,
                "ts": datetime.now(timezone.utc),
            },
        )
        row = result.fetchone()
        event_id = row[0] if row else None
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.error(f"Failed to insert feedback event: {exc}")
        raise HTTPException(status_code=500, detail="Failed to store feedback event.")

    try:
        invalidate_user_cache(body.user_id)
    except Exception as exc:
        logger.warning(f"Cache invalidation failed for user {body.user_id}: {exc}")

    logger.info(
        f"Feedback recorded: user={body.user_id} movie={body.movie_id} "
        f"type={body.event_type} event_id={event_id}"
    )

    return FeedbackResponse(
        success=True,
        message="Feedback event recorded successfully.",
        event_id=event_id,
    )
