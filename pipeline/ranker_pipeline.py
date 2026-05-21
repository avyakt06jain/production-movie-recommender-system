"""
Stage 2: Ranker Pipeline — wraps LightGBM ranker + feature construction.

Takes ~200 candidates from Stage 1, builds the full feature vector for each
(user, candidate) pair, runs LightGBM prediction, and returns top_k scored items.
"""

import re
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb

from training.dataset import (
    _encode_age_bucket,
    _encode_gender,
    N_GENRES,
)


def _is_sequel(title: str) -> int:
    """Heuristic: check if a movie title suggests it is a sequel."""
    if not title:
        return 0
    roman_pattern = r'\b(II|III|IV|VI{0,3}|IX|X)\b'
    digit_pattern = r'(?<!\()\b[2-9]\b(?!\d{3}\))'
    if re.search(roman_pattern, title) or re.search(digit_pattern, title):
        return 1
    return 0


class RankerPipeline:
    """
    Stage 2 of the recommendation pipeline.

    Takes candidate (movie_id, two_tower_score) pairs from Stage 1,
    constructs the full ~45-dim feature vector for each (user, item) pair,
    runs LightGBM prediction, and returns the top_k items by score.
    """

    def __init__(self, ranker_model: lgb.Booster, feature_store: dict):
        """
        Args:
            ranker_model:  Trained LightGBM Booster (lambdarank)
            feature_store: Features dict with user_features and item_features sub-dicts.
                           Also must contain popularity_percentiles and optionally
                           user_genre_rated_counts.
        """
        self.ranker = ranker_model
        self.feature_store = feature_store

        # Pre-compute popularity percentiles if not already present
        if "popularity_percentiles" not in self.feature_store:
            self._compute_popularity_percentiles()

    def _compute_popularity_percentiles(self) -> None:
        """Compute and cache popularity percentiles for all items."""
        item_features = self.feature_store.get("item_features", {})
        counts = []
        for iid, ifeat in item_features.items():
            counts.append((iid, ifeat.get("rating_count", 0)))
        counts.sort(key=lambda x: x[1])

        percentiles = {}
        n = max(len(counts) - 1, 1)
        for rank, (iid, _) in enumerate(counts):
            percentiles[iid] = rank / n

        self.feature_store["popularity_percentiles"] = percentiles

    def _build_feature_vector(
        self,
        user_id: int,
        item_id: int,
        two_tower_score: float,
    ) -> np.ndarray:
        """
        Build the full ranking feature vector for a (user, item) pair.

        Feature layout (49 dims total):
        [0]       user_age_norm
        [1]       user_gender_enc
        [2]       user_occupation
        [3]       user_avg_rating_given
        [4]       user_rating_count
        [5:23]    user_top_genres (18-dim)
        [23]      item_avg_rating
        [24]      item_rating_count_log
        [25]      item_year_norm
        [26:44]   item_genres (18-dim multi-hot)
        [44]      two_tower_score
        [45]      genre_overlap
        [46]      user_has_rated_genre
        [47]      popularity_percentile
        [48]      is_sequel
        """
        user_features = self.feature_store.get("user_features", {})
        item_features = self.feature_store.get("item_features", {})
        popularity_percentiles = self.feature_store.get("popularity_percentiles", {})

        uf = user_features.get(user_id, {})
        ifeat = item_features.get(item_id, {})

        if not uf or not ifeat:
            # Return zeros for unknown user/item
            return np.zeros(49, dtype=np.float32)

        # User features
        user_age_norm = _encode_age_bucket(uf.get("age", 25)) / 7.0
        user_gender_enc = 1 if uf.get("gender", "") == "M" else 0
        user_occupation = uf.get("occupation", 0)
        user_avg_rating = uf.get("avg_rating", 3.5)
        user_rating_count = uf.get("rating_count", 0)
        user_genre_vec = np.array(uf.get("watched_genre_vec", np.zeros(N_GENRES)), dtype=np.float32)

        # Item features
        item_avg_rating = ifeat.get("avg_rating", 3.0)
        item_log_count = ifeat.get("log_count", 0.0)
        item_year_norm = ifeat.get("year_norm", 0.5)
        item_genre_vec = np.array(ifeat.get("genre_vec", np.zeros(N_GENRES)), dtype=np.float32)

        # Interaction features
        genre_overlap = float(np.dot(user_genre_vec, item_genre_vec))

        # user_has_rated_genre: check if user has likely rated ≥3 movies in this genre
        user_has_rated_genre = 0
        rc = uf.get("rating_count", 0)
        for g_idx in range(N_GENRES):
            if item_genre_vec[g_idx] > 0:
                estimated_genre_count = int(user_genre_vec[g_idx] * rc)
                if estimated_genre_count >= 3:
                    user_has_rated_genre = 1
                    break

        popularity_pct = popularity_percentiles.get(item_id, 0.5)
        is_sequel = _is_sequel(ifeat.get("title", ""))

        features = np.concatenate([
            [user_age_norm, user_gender_enc, user_occupation, user_avg_rating, user_rating_count],
            user_genre_vec,
            [item_avg_rating, item_log_count, item_year_norm],
            item_genre_vec,
            [two_tower_score, genre_overlap, user_has_rated_genre, popularity_pct, is_sequel],
        ]).astype(np.float32)

        return features

    def rank(
        self,
        user_id: int,
        candidate_ids: list[int],
        two_tower_scores: dict[int, float],
        top_k: int = 20,
    ) -> list[dict]:
        """
        Rank candidate items for a user using LightGBM.

        Args:
            user_id:           Target user
            candidate_ids:     List of candidate movie IDs (from Stage 1)
            two_tower_scores:  Dict mapping movie_id → two-tower cosine score
            top_k:             Number of top-ranked items to return

        Returns:
            List of dicts sorted by descending ranker score:
            [{"movie_id": int, "score": float, "two_tower_score": float}, ...]
        """
        if not candidate_ids:
            return []

        # Build feature matrix for all candidates
        features_list = []
        valid_ids = []

        for mid in candidate_ids:
            tt_score = two_tower_scores.get(mid, 0.0)
            feat = self._build_feature_vector(user_id, mid, tt_score)
            features_list.append(feat)
            valid_ids.append(mid)

        if not features_list:
            return []

        features_matrix = np.array(features_list, dtype=np.float32)

        # Run LightGBM prediction
        scores = self.ranker.predict(features_matrix)

        # Sort by score descending
        scored_items = []
        for idx, mid in enumerate(valid_ids):
            scored_items.append({
                "movie_id": mid,
                "score": float(scores[idx]),
                "two_tower_score": two_tower_scores.get(mid, 0.0),
            })

        scored_items.sort(key=lambda x: x["score"], reverse=True)

        return scored_items[:top_k]
