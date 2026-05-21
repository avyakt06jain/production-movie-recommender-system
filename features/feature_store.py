"""
Feature Store for the MovieRec Recommender System
===================================================
Manages precomputed features (persisted in ``artifacts/features.pkl``) and
provides methods to build tensors for the Two-Tower model and feature vectors
for the LightGBM ranker.

See spec sections 6.3 and 8 for the full feature definitions.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENRE_LIST: list[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

NUM_GENRES: int = len(GENRE_LIST)  # 18

GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GENRE_LIST)}

# MovieLens 1M age bucket mapping (raw age code → ordinal bucket index)
AGE_BUCKET_MAP: dict[int, int] = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

# Gender encoding
GENDER_MAP: dict[str, int] = {"F": 0, "M": 1}

# Sequel heuristic regex: title contains a digit sequence or Roman numerals II/III/IV/V
_SEQUEL_RE = re.compile(r"\b(II|III|IV|V|VI|VII|VIII|IX|X|\d+)\b")

# Default artifact path (relative to project root)
DEFAULT_FEATURES_PATH = os.path.join("artifacts", "features.pkl")


class FeatureStore:
    """Central feature store for both training and inference.

    Attributes
    ----------
    user_features : dict[int, dict]
        Mapping from ``user_id`` to a dict with keys:
        ``genre_pref_vec``, ``avg_rating``, ``rating_count``,
        ``age_bucket``, ``gender``, ``occupation``,
        ``rated_movie_ids``, ``rating_timestamps``.
    item_features : dict[int, dict]
        Mapping from ``movie_id`` to a dict with keys:
        ``genre_vec``, ``year_norm``, ``avg_rating``, ``rating_count_log``,
        ``popularity_pct``, ``title``, ``genres``, ``year``, ``rating_count``.
    """

    def __init__(self) -> None:
        self.user_features: dict[int, dict[str, Any]] = {}
        self.item_features: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = DEFAULT_FEATURES_PATH) -> None:
        """Serialize both feature dicts to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_features": self.user_features,
            "item_features": self.item_features,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str = DEFAULT_FEATURES_PATH) -> None:
        """Load precomputed features from a pickle file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.user_features = payload["user_features"]
        self.item_features = payload["item_features"]

    # ------------------------------------------------------------------
    # Two-Tower model helpers
    # ------------------------------------------------------------------
    def get_user_features(self, user_id: int) -> dict[str, Any]:
        """Return user features as a dict of values ready for the UserTower.

        Returns a dict with keys:
        - ``user_id``: int
        - ``age_bucket``: int (0–6)
        - ``gender``: int (0 or 1)
        - ``occupation``: int (0–20)
        - ``watched_genre_vec``: np.ndarray of shape ``(18,)``
        """
        uf = self.user_features.get(user_id)
        if uf is None:
            # Unknown user → return neutral defaults
            return {
                "user_id": user_id,
                "age_bucket": 0,
                "gender": 0,
                "occupation": 0,
                "watched_genre_vec": np.zeros(NUM_GENRES, dtype=np.float32),
            }
        return {
            "user_id": user_id,
            "age_bucket": int(uf["age_bucket"]),
            "gender": int(uf["gender"]),
            "occupation": int(uf["occupation"]),
            "watched_genre_vec": np.asarray(uf["genre_pref_vec"], dtype=np.float32),
        }

    def get_item_features(self, movie_id: int) -> dict[str, Any]:
        """Return item features as a dict of values ready for the ItemTower.

        Returns a dict with keys:
        - ``item_id``: int
        - ``genre_vec``: np.ndarray of shape ``(18,)``
        - ``year_norm``: float
        - ``avg_rating_norm``: float (rating / 5.0)
        - ``log_count``: float (log1p of rating_count)
        """
        itf = self.item_features.get(movie_id)
        if itf is None:
            return {
                "item_id": movie_id,
                "genre_vec": np.zeros(NUM_GENRES, dtype=np.float32),
                "year_norm": 0.0,
                "avg_rating_norm": 0.0,
                "log_count": 0.0,
            }
        return {
            "item_id": movie_id,
            "genre_vec": np.asarray(itf["genre_vec"], dtype=np.float32),
            "year_norm": float(itf["year_norm"]),
            "avg_rating_norm": float(itf["avg_rating"]) / 5.0 if itf["avg_rating"] else 0.0,
            "log_count": float(itf["rating_count_log"]),
        }

    # ------------------------------------------------------------------
    # LightGBM ranking feature builder
    # ------------------------------------------------------------------
    def build_rank_features(
        self,
        user_id: int,
        candidate_ids: list[int],
        two_tower_scores: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Build the ~45 feature matrix for LightGBM ranking.

        Parameters
        ----------
        user_id : int
            Target user.
        candidate_ids : list[int]
            Movie IDs to score.
        two_tower_scores : dict[int, float], optional
            Mapping movie_id → cosine similarity from Stage 1.
            If ``None``, the ``two_tower_score`` feature is set to 0.

        Returns
        -------
        np.ndarray
            Shape ``(len(candidate_ids), ~45)`` float32 matrix.

        Feature layout (per row)::

            USER (23 features):
              [0]     user_age_norm
              [1]     user_gender_enc
              [2]     user_occupation
              [3]     user_avg_rating_given
              [4]     user_rating_count
              [5:23]  user_top_genres (18-dim)

            ITEM (21 features):
              [23]    item_avg_rating
              [24]    item_rating_count_log
              [25]    item_year_norm
              [26:44] item_genres (18-dim)

            INTERACTION (6 features):
              [44]    two_tower_score
              [45]    genre_overlap
              [46]    user_has_rated_genre
              [47]    rating_gap_years
              [48]    popularity_percentile
              [49]    is_sequel
        """
        if two_tower_scores is None:
            two_tower_scores = {}

        n = len(candidate_ids)
        n_features = 23 + 21 + 6  # 50
        features = np.zeros((n, n_features), dtype=np.float32)

        # Fetch user data
        uf = self.user_features.get(user_id, {})
        user_age_norm = self._age_bucket_to_norm(uf.get("age_bucket", 0))
        user_gender_enc = int(uf.get("gender", 0))
        user_occupation = int(uf.get("occupation", 0))
        user_avg_rating = float(uf.get("avg_rating", 3.0))
        user_rating_count = int(uf.get("rating_count", 0))
        user_genre_vec = np.asarray(
            uf.get("genre_pref_vec", np.zeros(NUM_GENRES)), dtype=np.float32
        )

        # User's rated movie set and timestamps (for interaction features)
        user_rated_ids: set[int] = set(uf.get("rated_movie_ids", []))
        user_rating_ts: dict[int, int] = uf.get("rating_timestamps", {})

        # Precompute the set of genre indices the user has rated ≥3 movies in
        user_genre_counts = np.zeros(NUM_GENRES, dtype=np.int32)
        for mid in user_rated_ids:
            itf = self.item_features.get(mid)
            if itf is not None:
                gv = np.asarray(itf["genre_vec"], dtype=np.float32)
                user_genre_counts += (gv > 0).astype(np.int32)

        # Earliest rating timestamp per genre for this user
        earliest_genre_ts: dict[int, int] = {}
        for mid in user_rated_ids:
            itf = self.item_features.get(mid)
            ts = user_rating_ts.get(mid)
            if itf is None or ts is None:
                continue
            gv = np.asarray(itf["genre_vec"], dtype=np.float32)
            for gi in range(NUM_GENRES):
                if gv[gi] > 0:
                    if gi not in earliest_genre_ts or ts < earliest_genre_ts[gi]:
                        earliest_genre_ts[gi] = ts

        # Latest timestamp across all of this user's ratings (for gap calculation)
        if user_rating_ts:
            user_latest_ts = max(user_rating_ts.values())
        else:
            user_latest_ts = 0

        # Fill feature matrix
        for i, movie_id in enumerate(candidate_ids):
            itf = self.item_features.get(movie_id, {})
            item_genre_vec = np.asarray(
                itf.get("genre_vec", np.zeros(NUM_GENRES)), dtype=np.float32
            )

            # ----- USER features (0..22) -----
            features[i, 0] = user_age_norm
            features[i, 1] = user_gender_enc
            features[i, 2] = user_occupation
            features[i, 3] = user_avg_rating
            features[i, 4] = user_rating_count
            features[i, 5:23] = user_genre_vec

            # ----- ITEM features (23..43) -----
            features[i, 23] = float(itf.get("avg_rating", 0.0) or 0.0)
            features[i, 24] = float(itf.get("rating_count_log", 0.0))
            features[i, 25] = float(itf.get("year_norm", 0.0))
            features[i, 26:44] = item_genre_vec

            # ----- INTERACTION features (44..49) -----
            # two_tower_score
            features[i, 44] = two_tower_scores.get(movie_id, 0.0)

            # genre_overlap = dot(user_genre_vec, item_genre_vec)
            features[i, 45] = float(np.dot(user_genre_vec, item_genre_vec))

            # user_has_rated_genre: 1 if user rated ≥3 movies in ANY of the item's genres
            item_genre_indices = np.where(item_genre_vec > 0)[0]
            has_rated_genre = 0
            for gi in item_genre_indices:
                if user_genre_counts[gi] >= 3:
                    has_rated_genre = 1
                    break
            features[i, 46] = has_rated_genre

            # rating_gap_years: years since user first rated a movie of a similar genre
            if item_genre_indices.size > 0 and earliest_genre_ts:
                earliest_for_item = min(
                    (earliest_genre_ts.get(gi, user_latest_ts) for gi in item_genre_indices),
                    default=user_latest_ts,
                )
                gap_seconds = max(0, user_latest_ts - earliest_for_item)
                features[i, 47] = gap_seconds / (365.25 * 24 * 3600)
            else:
                features[i, 47] = 0.0

            # popularity_percentile
            features[i, 48] = float(itf.get("popularity_pct", 0.0))

            # is_sequel: heuristic — title contains a digit or Roman numeral
            title = itf.get("title", "")
            features[i, 49] = 1.0 if _SEQUEL_RE.search(title) else 0.0

        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _age_bucket_to_norm(bucket: int) -> float:
        """Normalize age bucket (0–6) to [0, 1]."""
        return bucket / 6.0

    @staticmethod
    def genres_to_vec(genres: list[str]) -> np.ndarray:
        """Convert a list of genre strings to a multi-hot 18-dim vector."""
        vec = np.zeros(NUM_GENRES, dtype=np.float32)
        for g in genres:
            idx = GENRE_TO_IDX.get(g)
            if idx is not None:
                vec[idx] = 1.0
        return vec
