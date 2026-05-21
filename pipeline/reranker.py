"""
Stage 3: Re-ranker with MMR (Maximal Marginal Relevance) diversity filter.

Takes the top-20 from Stage 2 and applies:
1. Already-watched filter — removes movies the user has already rated
2. MMR re-ranking — balances relevance and diversity using genre vectors
3. Returns the final top-N recommendations with full metadata
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a, b: 1-D numpy arrays (e.g., 18-dim genre vectors)

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_rerank(
    candidates: list[dict],
    top_n: int = 10,
    lambda_mmr: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance re-ranking for diversity.

    Iteratively selects items that maximize:
        λ * relevance_score - (1-λ) * max_similarity_to_already_selected

    This balances relevance (high λ) and diversity (low λ) in the final
    recommendation list.

    Args:
        candidates: List of dicts, each must have:
                    - "score": float (relevance from ranker)
                    - "genre_vec": np.ndarray (18-dim genre vector)
        top_n:      Number of items to select
        lambda_mmr: Trade-off parameter (0=pure diversity, 1=pure relevance)

    Returns:
        List of top_n selected candidates in MMR order
    """
    if not candidates:
        return []

    if len(candidates) <= top_n:
        return list(candidates)

    # Normalize scores to [0, 1] for fair weighting
    scores = [c["score"] for c in candidates]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score
    if score_range == 0:
        score_range = 1.0

    selected: list[dict] = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        if not selected:
            # First pick: highest relevance score
            best = max(remaining, key=lambda x: x["score"])
        else:
            # MMR selection
            best = None
            best_mmr = float("-inf")

            for item in remaining:
                # Normalized relevance
                rel = (item["score"] - min_score) / score_range

                # Max similarity to any already-selected item
                item_genre = np.array(item["genre_vec"], dtype=np.float32)
                max_sim = max(
                    cosine_sim(item_genre, np.array(s["genre_vec"], dtype=np.float32))
                    for s in selected
                )

                mmr_val = lambda_mmr * rel - (1 - lambda_mmr) * max_sim

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best = item

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)

    return selected


class Reranker:
    """
    Stage 3 of the recommendation pipeline.

    Applies watched-movie filtering and MMR diversity re-ranking to produce
    the final top-N recommendation list with full metadata.
    """

    def __init__(self, feature_store: dict):
        """
        Args:
            feature_store: Features dict with item_features and user_all_items sub-dicts
        """
        self.feature_store = feature_store

    def rerank(
        self,
        user_id: int,
        ranked_candidates: list[dict],
        top_n: int = 10,
        lambda_mmr: float = 0.7,
        exclude_watched: bool = True,
    ) -> list[dict]:
        """
        Apply post-ranking filters and diversity re-ranking.

        Args:
            user_id:           Target user
            ranked_candidates: List of dicts from Stage 2, each with at least:
                               - "movie_id": int
                               - "score": float
                               - "two_tower_score": float (optional)
            top_n:             Number of final recommendations
            lambda_mmr:        MMR diversity parameter (0.7 = favor relevance)
            exclude_watched:   Whether to filter already-watched movies

        Returns:
            List of final recommendation dicts with full metadata:
            [{
                "movie_id": int,
                "title": str,
                "genres": list[str],
                "two_tower_score": float,
                "ranker_score": float,
                "final_rank": int,
            }, ...]
        """
        item_features = self.feature_store.get("item_features", {})
        user_all_items = self.feature_store.get("user_all_items", {})

        # ── Step 1: Filter already-watched movies ──────────────────────────
        if exclude_watched:
            watched = user_all_items.get(user_id, set())
            candidates = [c for c in ranked_candidates if c["movie_id"] not in watched]
            n_filtered = len(ranked_candidates) - len(candidates)
            if n_filtered > 0:
                logger.debug(f"Filtered {n_filtered} already-watched movies for user {user_id}")
        else:
            candidates = list(ranked_candidates)

        if not candidates:
            logger.warning(f"No candidates remaining for user {user_id} after filtering")
            return []

        # ── Step 2: Enrich with genre vectors for MMR ──────────────────────
        enriched_candidates = []
        for c in candidates:
            mid = c["movie_id"]
            ifeat = item_features.get(mid, {})
            genre_vec = ifeat.get("genre_vec", np.zeros(18))

            enriched_candidates.append({
                **c,
                "genre_vec": np.array(genre_vec, dtype=np.float32),
            })

        # ── Step 3: MMR re-ranking ─────────────────────────────────────────
        reranked = mmr_rerank(enriched_candidates, top_n=top_n, lambda_mmr=lambda_mmr)

        # ── Step 4: Build final output with metadata ───────────────────────
        results = []
        for rank, item in enumerate(reranked, start=1):
            mid = item["movie_id"]
            ifeat = item_features.get(mid, {})

            results.append({
                "movie_id": mid,
                "title": ifeat.get("title", f"Movie {mid}"),
                "genres": ifeat.get("genres", []),
                "two_tower_score": round(item.get("two_tower_score", 0.0), 4),
                "ranker_score": round(item.get("score", 0.0), 4),
                "final_rank": rank,
            })

        return results
