"""
GET /api/v1/recommend/{user_id}

Full 3-stage recommendation pipeline:
  1. Cache check  →  hit ⇒ return immediately
  2. Candidate generation (Two-Tower user embedding → FAISS ANN → top-200)
  3. Ranking (LightGBM scores top-200 → keep top-20)
  4. Re-ranking (already-watched filter + MMR diversity → final top-N)
"""

import os
import time
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from api.cache import TTL_RECS, get_cached, set_cached

router = APIRouter(tags=["recommendations"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


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
    """Maximal Marginal Relevance re-ranking for diversity.

    Each candidate must have keys: ``score`` (float) and ``genre_vec`` (np array).
    ``lambda_mmr`` blends relevance vs diversity (1.0 = pure relevance).
    """
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
    """Build the (N, F) feature matrix for LightGBM ranking.

    For each (user, candidate_item) pair we concatenate:
      - user features (age_norm, gender_enc, occupation, avg_rating, rating_count, genre_pref 18-d)
      - item features (avg_rating, rating_count_log, year_norm, genre_vec 18-d)
      - interaction features (two_tower_score, genre_overlap, popularity_pct)
    Total: ~45 features
    """
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

        # Interaction features
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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
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

    # ── Validate user exists ──────────────────────────────────────────
    if registry.feature_store is None or user_id not in registry.feature_store.user_features:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in the system.")

    # ── Stage 0: Cache check ──────────────────────────────────────────
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

    # ── Stage 1: Candidate generation (Two-Tower → FAISS) ────────────
    user_feats = registry.feature_store.user_features[user_id]

    if registry.two_tower is None or registry.faiss_index is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check /health for status.",
        )

    faiss_top_k = int(os.environ.get("FAISS_TOP_K", 200))
    user_batch = _build_user_tensor(user_feats, registry)

    with torch.no_grad():
        user_emb = registry.two_tower.user_tower(**user_batch)  # (1, 64)

    user_emb_np = user_emb.numpy().astype(np.float32)

    # FAISS search returns (scores, indices); we need the IDs + scores
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

    # ── Stage 2: Ranking (LightGBM) ──────────────────────────────────
    ranker_top_k = int(os.environ.get("RANKER_TOP_K", 20))

    if registry.ranker is not None and candidate_ids:
        rank_features = _build_rank_features(
            user_feats, candidate_ids, candidate_scores, registry
        )
        lgbm_scores = registry.ranker.predict(rank_features)

        # Sort by LightGBM score descending → keep top-K
        scored_pairs = sorted(
            zip(candidate_ids, lgbm_scores), key=lambda x: x[1], reverse=True
        )
        candidate_ids = [mid for mid, _ in scored_pairs[:ranker_top_k]]
        candidate_scores = {mid: float(sc) for mid, sc in scored_pairs[:ranker_top_k]}
    else:
        # Fallback: use two-tower scores and just take the top-K
        scored_pairs = sorted(
            ((mid, candidate_scores.get(mid, 0.0)) for mid in candidate_ids),
            key=lambda x: x[1],
            reverse=True,
        )
        candidate_ids = [mid for mid, _ in scored_pairs[:ranker_top_k]]
        candidate_scores = {mid: sc for mid, sc in scored_pairs[:ranker_top_k]}

    # ── Stage 3: Re-ranking (watched filter + MMR) ────────────────────
    # Build watched set from the feature store (user's rated movies)
    watched_set: set[int] = set()
    if exclude_watched:
        watched_set = set(user_feats.get("rated_movie_ids", []))

    # Normalise scores to [0, 1] for MMR
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

    # ── Build response ────────────────────────────────────────────────
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

    # ── Cache the result ──────────────────────────────────────────────
    set_cached(cache_key, recommendations, TTL_RECS)

    latency = (time.perf_counter() - t0) * 1000
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[RecommendationItem(**r) for r in recommendations],
        latency_ms=round(latency, 1),
        cache_hit=False,
    )

