#!/usr/bin/env python3
"""
Feature Precomputation Script
==============================
Connects to PostgreSQL, loads all users/movies/ratings, computes every
precomputed feature described in spec section 8.1, and saves them to
``artifacts/features.pkl`` via the :class:`FeatureStore`.

Run from the project root:
    python scripts/precompute_features.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from models.feature_store import (
    AGE_BUCKET_MAP,
    GENDER_MAP,
    GENRE_LIST,
    GENRE_TO_IDX,
    NUM_GENRES,
    FeatureStore,
)


# ---------------------------------------------------------------------------
# Data loading helpers (sync via psycopg2)
# ---------------------------------------------------------------------------
def load_all_data(conn) -> tuple[list, list, list]:
    """Load users, movies, and ratings from the database.

    Handles both PostgreSQL (genres as TEXT[]) and SQLite (genres as pipe-separated string).

    Returns
    -------
    users : list[tuple]
        Each tuple is (user_id, gender, age, occupation, zip_code).
    movies : list[tuple]
        Each tuple is (movie_id, title, year, genres_list, avg_rating, rating_count).
    ratings : list[tuple]
        Each tuple is (user_id, movie_id, rating, timestamp).
    """
    from api.database import is_sqlite
    use_sqlite = is_sqlite(conn)

    cur = conn.cursor()

    cur.execute("SELECT user_id, gender, age, occupation, zip_code FROM users ORDER BY user_id")
    users = [tuple(row) for row in cur.fetchall()]
    logger.info(f"  Loaded {len(users):,} users from DB.")

    cur.execute(
        "SELECT movie_id, title, year, genres, avg_rating, rating_count "
        "FROM movies ORDER BY movie_id"
    )
    raw_movies = cur.fetchall()
    movies = []
    for row in raw_movies:
        row = tuple(row)
        movie_id, title, year, genres_raw, avg_rating, rating_count = row
        # Handle genres: SQLite stores as pipe-separated string, PostgreSQL as array
        if isinstance(genres_raw, str):
            genres_list = genres_raw.split("|") if genres_raw else []
        elif isinstance(genres_raw, list):
            genres_list = genres_raw
        else:
            genres_list = []
        movies.append((movie_id, title, year, genres_list, avg_rating, rating_count))
    logger.info(f"  Loaded {len(movies):,} movies from DB.")

    cur.execute(
        "SELECT user_id, movie_id, rating, timestamp FROM ratings ORDER BY user_id, timestamp"
    )
    ratings = [tuple(row) for row in cur.fetchall()]
    logger.info(f"  Loaded {len(ratings):,} ratings from DB.")

    return users, movies, ratings


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_item_features(
    movies: list[tuple],
) -> dict[int, dict]:
    """Compute all per-item precomputed features.

    Features per item:
    - ``genre_vec``: 18-dim multi-hot
    - ``year_norm``: (year - 1920) / 100
    - ``avg_rating``: global mean (already in DB, but we also store here)
    - ``rating_count_log``: log1p(rating_count)
    - ``popularity_pct``: percentile rank by rating_count
    - ``title``, ``genres``, ``year``, ``rating_count``: raw metadata
    """
    # First pass: collect rating counts for percentile computation
    counts = []
    for movie_id, title, year, genres, avg_rating, rating_count in movies:
        counts.append(rating_count or 0)
    counts_arr = np.array(counts, dtype=np.float64)

    # Compute percentile for each movie
    # percentile = fraction of movies with rating_count <= this movie's count
    sorted_counts = np.sort(counts_arr)
    n_movies = len(sorted_counts)

    item_features: dict[int, dict] = {}
    for idx, (movie_id, title, year, genres, avg_rating, rating_count) in enumerate(movies):
        rc = rating_count or 0

        # Genre vector
        genres_list = genres if genres else []
        genre_vec = np.zeros(NUM_GENRES, dtype=np.float32)
        for g in genres_list:
            gi = GENRE_TO_IDX.get(g)
            if gi is not None:
                genre_vec[gi] = 1.0

        # Year normalization
        year_norm = (year - 1920) / 100.0 if year else 0.0

        # Popularity percentile: fraction of movies with count <= rc
        pct = float(np.searchsorted(sorted_counts, rc, side="right")) / n_movies

        item_features[movie_id] = {
            "genre_vec": genre_vec,
            "year_norm": year_norm,
            "avg_rating": avg_rating if avg_rating else 0.0,
            "rating_count_log": float(np.log1p(rc)),
            "popularity_pct": pct,
            "title": title,
            "genres": genres_list,
            "year": year,
            "rating_count": rc,
        }

    return item_features


def compute_user_features(
    users: list[tuple],
    ratings: list[tuple],
    item_features: dict[int, dict],
) -> dict[int, dict]:
    """Compute all per-user precomputed features.

    Features per user:
    - ``genre_pref_vec``: 18-dim avg genre vector of positively-rated (≥4) movies
    - ``avg_rating``: mean of all ratings given
    - ``rating_count``: total ratings
    - ``age_bucket``: ordinal bucket (0–6) from the MovieLens age mapping
    - ``gender``: 0 (F) or 1 (M)
    - ``occupation``: 0–20
    - ``rated_movie_ids``: set of movie_ids the user has rated
    - ``rating_timestamps``: dict mapping movie_id → timestamp
    """
    # Build per-user data structures
    user_ratings: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for user_id, movie_id, rating, timestamp in ratings:
        user_ratings[user_id].append((movie_id, rating, timestamp))

    # User metadata lookup
    user_meta: dict[int, tuple] = {}
    for user_id, gender, age, occupation, zip_code in users:
        user_meta[user_id] = (gender, age, occupation)

    user_features: dict[int, dict] = {}
    for user_id, gender, age, occupation, zip_code in users:
        u_ratings = user_ratings.get(user_id, [])

        # Average rating
        if u_ratings:
            all_ratings = [r for _, r, _ in u_ratings]
            avg_rating = float(np.mean(all_ratings))
            rating_count = len(all_ratings)
        else:
            avg_rating = 0.0
            rating_count = 0

        # Genre preference vector: average genre vector of positively-rated movies (rating >= 4)
        pos_genre_vecs = []
        for mid, r, _ in u_ratings:
            if r >= 4.0 and mid in item_features:
                pos_genre_vecs.append(item_features[mid]["genre_vec"])

        if pos_genre_vecs:
            genre_pref_vec = np.mean(pos_genre_vecs, axis=0).astype(np.float32)
        else:
            genre_pref_vec = np.zeros(NUM_GENRES, dtype=np.float32)

        # Rated movie IDs and timestamps
        rated_movie_ids = [mid for mid, _, _ in u_ratings]
        rating_timestamps = {mid: ts for mid, _, ts in u_ratings}

        # Age bucket mapping
        age_bucket = AGE_BUCKET_MAP.get(age, 0)

        # Gender encoding
        gender_enc = GENDER_MAP.get(gender, 0)

        user_features[user_id] = {
            "user_id": user_id,
            "genre_pref_vec": genre_pref_vec,
            "watched_genre_vec": genre_pref_vec,  # alias for dataset.py compatibility
            "avg_rating": avg_rating,
            "rating_count": rating_count,
            "age_bucket": age_bucket,
            "age": age,              # raw MovieLens age code — needed by dataset.py
            "gender": gender_enc,
            "gender_raw": gender,    # raw string — needed by dataset.py
            "occupation": occupation,
            "rated_movie_ids": rated_movie_ids,
            "rating_timestamps": rating_timestamps,
        }

    return user_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Feature Precomputation Script")
    logger.info("=" * 60)

    # Connect to DB
    from api.database import get_sync_connection

    logger.info("Connecting to PostgreSQL ...")
    conn = get_sync_connection()

    try:
        logger.info("Loading data from database ...")
        users, movies, ratings = load_all_data(conn)
    finally:
        conn.close()

    if not users or not movies or not ratings:
        logger.error("No data found in database. Run scripts/load_data.py first.")
        sys.exit(1)

    # Compute features
    logger.info("Computing item features ...")
    item_feats = compute_item_features(movies)
    logger.info(f"  Computed features for {len(item_feats):,} items.")

    logger.info("Computing user features ...")
    user_feats = compute_user_features(users, ratings, item_feats)
    logger.info(f"  Computed features for {len(user_feats):,} users.")

    # Build training data structures needed by dataset.py
    logger.info("Building training data structures ...")
    import pandas as pd

    # user_positive_items: user_id -> list of movie_ids rated >= 4.0
    user_positive_items: dict[int, list[int]] = {}
    # user_all_items: user_id -> set of all rated movie_ids
    user_all_items: dict[int, set[int]] = {}

    for uid, uf in user_feats.items():
        user_all_items[uid] = set(uf.get("rated_movie_ids", []))

    # Build from raw ratings for accuracy
    from collections import defaultdict as _defaultdict
    _user_pos = _defaultdict(list)
    for user_id, movie_id, rating, timestamp in ratings:
        if rating >= 4.0 and movie_id in item_feats:
            _user_pos[user_id].append(movie_id)
    user_positive_items = dict(_user_pos)

    all_item_ids = sorted(item_feats.keys())

    # Build ratings DataFrame for time-based splitting
    ratings_df = pd.DataFrame(ratings, columns=["user_id", "movie_id", "rating", "timestamp"])

    logger.info(f"  user_positive_items: {len(user_positive_items):,} users with positive items")
    logger.info(f"  all_item_ids: {len(all_item_ids):,} items")

    # Build and save FeatureStore
    fs = FeatureStore()
    fs.user_features = user_feats
    fs.item_features = item_feats
    fs.training_data = {
        "user_positive_items": user_positive_items,
        "user_all_items": user_all_items,
        "all_item_ids": all_item_ids,
        "ratings_df": ratings_df,
    }

    output_path = str(PROJECT_ROOT / "artifacts" / "features.pkl")
    logger.info(f"Saving feature store to {output_path} ...")
    fs.save(output_path)

    # Verification
    import os as _os

    file_size_mb = _os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Quick sanity check: reload and verify
    fs2 = FeatureStore()
    fs2.load(output_path)
    assert len(fs2.user_features) == len(user_feats), "User feature count mismatch!"
    assert len(fs2.item_features) == len(item_feats), "Item feature count mismatch!"

    # Sample feature for first user
    sample_uid = next(iter(fs2.user_features))
    sample_uf = fs2.get_user_features(sample_uid)
    logger.info(f"  Sample user {sample_uid}: age_bucket={sample_uf['age_bucket']}, "
                f"genre_vec_sum={sample_uf['watched_genre_vec'].sum():.3f}")

    # Sample feature for first item
    sample_mid = next(iter(fs2.item_features))
    sample_if = fs2.get_item_features(sample_mid)
    logger.info(f"  Sample item {sample_mid}: year_norm={sample_if['year_norm']:.3f}, "
                f"genre_vec_sum={sample_if['genre_vec'].sum():.0f}")

    # Test rank feature builder
    sample_candidates = list(fs2.item_features.keys())[:5]
    rank_feats = fs2.build_rank_features(sample_uid, sample_candidates)
    logger.info(f"  Rank feature matrix shape: {rank_feats.shape} "
                f"(expected ({len(sample_candidates)}, 50))")

    logger.info("=" * 60)
    logger.info("Feature precomputation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
