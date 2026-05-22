#!/usr/bin/env python3
"""
Synthetic MovieLens 1M Data Generator
======================================
Generates realistic synthetic data matching the MovieLens 1M schema
when the real dataset cannot be downloaded.

Produces:
- 6,040 users with demographics
- 3,706 movies with titles, years, and genres
- ~1,000,000 ratings (1-5 stars)

All distributions are modeled after the real MovieLens 1M to ensure
the ML pipeline trains and evaluates properly.

Usage:
    python scripts/generate_synthetic_data.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants matching MovieLens 1M ──────────────────────────────────────────

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Genre weights (approximate real distribution)
GENRE_WEIGHTS = [0.12, 0.08, 0.04, 0.05, 0.20, 0.08, 0.03, 0.25,
                 0.03, 0.02, 0.07, 0.04, 0.05, 0.10, 0.07, 0.12, 0.04, 0.02]

AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]
AGE_WEIGHTS = [0.03, 0.20, 0.30, 0.22, 0.12, 0.08, 0.05]

OCCUPATIONS = list(range(21))

# Real movie title prefixes for realism
TITLE_WORDS = [
    "The", "A", "Dark", "Last", "Night", "Star", "Love", "Black", "Red",
    "Final", "Lost", "Secret", "Golden", "Silver", "Iron", "Steel",
    "Shadow", "Fire", "Ice", "Storm", "Thunder", "Silent", "Wild",
    "Dead", "American", "Good", "Bad", "Big", "Little", "Old", "New",
    "Return", "Rise", "Fall", "Day", "War", "Time", "City", "Road",
    "Man", "Woman", "King", "Queen", "Prince", "Knight", "Angel",
    "Devil", "Ghost", "Dream", "Mission", "Journey", "Legend",
]

TITLE_SUFFIXES = [
    "of Doom", "Returns", "Reloaded", "Unleashed", "Chronicles",
    "Story", "Adventures", "Legacy", "Rising", "Awakening",
    "Strikes Back", "Begins", "Forever", "Redemption", "Revolution",
    "", "", "", "", "",  # many movies have no suffix
]

N_USERS = 6040
N_MOVIES = 3706
TARGET_RATINGS = 1000000
MIN_YEAR = 1920
MAX_YEAR = 2000

# Timestamp range (mimics MovieLens 1M: April 2000 to Feb 2003)
TS_MIN = 956703932
TS_MAX = 1046454052


def generate_users(n: int = N_USERS) -> list[tuple]:
    """Generate synthetic users: (user_id, gender, age, occupation, zip_code)."""
    users = []
    for uid in range(1, n + 1):
        gender = random.choice(["M", "F"])
        age = random.choices(AGE_BUCKETS, weights=AGE_WEIGHTS, k=1)[0]
        occupation = random.choice(OCCUPATIONS)
        zip_code = f"{random.randint(10000, 99999)}"
        users.append((uid, gender, age, occupation, zip_code))
    return users


def generate_movies(n: int = N_MOVIES) -> list[tuple]:
    """Generate synthetic movies: (movie_id, title, year, genres_list)."""
    movies = []
    used_titles = set()

    for mid in range(1, n + 1):
        # Generate unique title
        while True:
            n_words = random.randint(1, 3)
            words = random.sample(TITLE_WORDS, n_words)
            suffix = random.choice(TITLE_SUFFIXES)
            title_base = " ".join(words)
            if suffix:
                title_base = f"{title_base}: {suffix}"
            if title_base not in used_titles:
                used_titles.add(title_base)
                break

        # Year distribution weighted toward recent
        year = int(np.random.triangular(MIN_YEAR, 1990, MAX_YEAR))
        title = f"{title_base} ({year})"

        # Genres: 1-3 genres per movie
        n_genres = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2], k=1)[0]
        genres = list(np.random.choice(
            GENRE_LIST, size=n_genres, replace=False, p=np.array(GENRE_WEIGHTS) / sum(GENRE_WEIGHTS)
        ))

        movies.append((mid, title, year, genres))

    return movies


def generate_ratings(
    users: list[tuple],
    movies: list[tuple],
    target_n: int = TARGET_RATINGS,
) -> list[tuple]:
    """Generate synthetic ratings: (user_id, movie_id, rating, timestamp).

    Models realistic user behavior:
    - Users have genre preferences that influence their ratings
    - Popular movies get more ratings (power-law distribution)
    - Rating values follow a realistic distribution (skewed toward 3-4)
    """
    n_users = len(users)
    n_movies = len(movies)

    # Movie popularity: power-law distribution
    movie_popularity = np.random.power(0.5, n_movies)
    movie_popularity = movie_popularity / movie_popularity.sum()

    # User activity: log-normal distribution (some users rate a lot)
    user_activity = np.random.lognormal(mean=4.5, sigma=1.0, size=n_users)
    user_activity = np.clip(user_activity, 20, 2000).astype(int)
    # Scale to hit target total
    total = user_activity.sum()
    user_activity = (user_activity * target_n / total).astype(int)

    # Genre preferences per user (soft preferences)
    user_genre_prefs = {}
    for uid, gender, age, occ, _ in users:
        prefs = np.random.dirichlet(np.ones(len(GENRE_LIST)) * 2.0)
        user_genre_prefs[uid] = prefs

    # Movie genre vectors
    movie_genres = {}
    for mid, title, year, genres in movies:
        vec = np.zeros(len(GENRE_LIST))
        for g in genres:
            vec[GENRE_LIST.index(g)] = 1.0
        movie_genres[mid] = vec

    ratings = []
    movie_ids = [m[0] for m in movies]

    for i, (uid, gender, age, occ, _) in enumerate(users):
        n_ratings = user_activity[i]
        if n_ratings < 1:
            n_ratings = 20

        # Weight movies by popularity * genre affinity
        genre_pref = user_genre_prefs[uid]
        affinities = np.array([
            np.dot(genre_pref, movie_genres[mid]) for mid in movie_ids
        ])
        weights = movie_popularity * (0.3 + 0.7 * affinities)
        weights = weights / weights.sum()

        # Sample movies
        rated_movies = np.random.choice(
            movie_ids, size=min(n_ratings, n_movies), replace=False, p=weights
        )

        for mid in rated_movies:
            # Rating: influenced by genre preference + random noise
            affinity = np.dot(genre_pref, movie_genres[mid])
            base_rating = 2.5 + affinity * 2.5  # 2.5 to 5.0 based on affinity
            noise = np.random.normal(0, 0.8)
            rating = np.clip(round(base_rating + noise), 1, 5)
            rating = float(rating)

            # Timestamp: random within range
            ts = random.randint(TS_MIN, TS_MAX)

            ratings.append((uid, mid, rating, ts))

    # Sort by timestamp for time-based splitting
    ratings.sort(key=lambda x: x[3])

    logger.info(f"Generated {len(ratings):,} ratings for {n_users:,} users across {n_movies:,} movies")
    return ratings


def save_as_dat_files(users, movies, ratings, output_dir: Path):
    """Save in MovieLens .dat format for compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # users.dat
    with open(output_dir / "users.dat", "w") as f:
        for uid, gender, age, occ, zip_code in users:
            f.write(f"{uid}::{gender}::{age}::{occ}::{zip_code}\n")

    # movies.dat
    with open(output_dir / "movies.dat", "w") as f:
        for mid, title, year, genres in movies:
            f.write(f"{mid}::{title}::{'|'.join(genres)}\n")

    # ratings.dat
    with open(output_dir / "ratings.dat", "w") as f:
        for uid, mid, rating, ts in ratings:
            f.write(f"{uid}::{mid}::{int(rating)}::{ts}\n")

    logger.info(f"Saved .dat files to {output_dir}")


def main():
    logger.info("=" * 60)
    logger.info("Synthetic MovieLens 1M Data Generator")
    logger.info("=" * 60)

    random.seed(42)
    np.random.seed(42)

    users = generate_users()
    logger.info(f"Generated {len(users):,} users")

    movies = generate_movies()
    logger.info(f"Generated {len(movies):,} movies")

    ratings = generate_ratings(users, movies)

    # Save .dat files
    dat_dir = PROJECT_ROOT / "data" / "raw" / "ml-1m"
    save_as_dat_files(users, movies, ratings, dat_dir)

    logger.info("=" * 60)
    logger.info("Synthetic data generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
