#!/usr/bin/env python3
"""
Data Loading Script
====================
Downloads MovieLens 1M, parses the raw files, and bulk-loads into the database.
Supports both PostgreSQL (production) and SQLite (local development).

Run from the project root:
    python scripts/load_data.py
"""

import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from api.database import get_sync_connection, is_sqlite


# ─── Constants ────────────────────────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR = DATA_DIR / "ml-1m"


# ─── SQL Schema ──────────────────────────────────────────────────────────────

POSTGRES_SCHEMA = """
DROP TABLE IF EXISTS feedback_events CASCADE;
DROP TABLE IF EXISTS ratings CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    gender      CHAR(1),
    age         INTEGER,
    occupation  INTEGER,
    zip_code    VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    year        INTEGER,
    genres      TEXT[],
    avg_rating  FLOAT,
    rating_count INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE ratings (
    rating_id   SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      FLOAT NOT NULL,
    timestamp   BIGINT,
    implicit    BOOLEAN GENERATED ALWAYS AS (rating >= 4.0) STORED,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ratings_user    ON ratings(user_id);
CREATE INDEX idx_ratings_movie   ON ratings(movie_id);
CREATE INDEX idx_ratings_implicit ON ratings(user_id, implicit);

CREATE TABLE feedback_events (
    event_id    SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    event_type  VARCHAR(20),
    rating      FLOAT,
    session_id  VARCHAR(64),
    timestamp   TIMESTAMP DEFAULT NOW()
);
"""

SQLITE_SCHEMA = """
DROP TABLE IF EXISTS feedback_events;
DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    gender      TEXT,
    age         INTEGER,
    occupation  INTEGER,
    zip_code    TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    year        INTEGER,
    genres      TEXT,
    avg_rating  REAL,
    rating_count INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE ratings (
    rating_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      REAL NOT NULL,
    timestamp   INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_ratings_user    ON ratings(user_id);
CREATE INDEX idx_ratings_movie   ON ratings(movie_id);

CREATE TABLE feedback_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    event_type  TEXT,
    rating      REAL,
    session_id  TEXT,
    timestamp   TEXT DEFAULT (datetime('now'))
);
"""


# ─── Download & extract ──────────────────────────────────────────────────────
def download_and_extract():
    """Download ml-1m.zip and extract to data/raw/ml-1m/.

    Falls back to synthetic data generation if download fails.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"

    # Check if data already exists
    if EXTRACT_DIR.exists() and (EXTRACT_DIR / "users.dat").exists():
        logger.info(f"Data already exists at {EXTRACT_DIR}")
        return

    # Try downloading
    if not zip_path.exists():
        try:
            logger.info(f"Downloading MovieLens 1M from {MOVIELENS_URL} ...")
            urllib.request.urlretrieve(MOVIELENS_URL, str(zip_path))
            logger.info(f"Downloaded to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            logger.info("Falling back to synthetic data generation ...")
            from scripts.generate_synthetic_data import main as generate_main
            generate_main()
            return

    if zip_path.exists():
        logger.info("Extracting ...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(DATA_DIR))
        logger.info(f"Extracted to {EXTRACT_DIR}")
    else:
        logger.info("No zip file found, generating synthetic data ...")
        from scripts.generate_synthetic_data import main as generate_main
        generate_main()


# ─── Parse raw .dat files ────────────────────────────────────────────────────
def parse_users():
    """Parse users.dat → list of (user_id, gender, age, occupation, zip_code)."""
    users = []
    path = EXTRACT_DIR / "users.dat"
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            users.append((int(parts[0]), parts[1], int(parts[2]), int(parts[3]), parts[4]))
    logger.info(f"Parsed {len(users):,} users from users.dat")
    return users


def parse_movies():
    """Parse movies.dat → list of (movie_id, title, year, genres_list)."""
    movies = []
    year_re = re.compile(r"\((\d{4})\)\s*$")
    path = EXTRACT_DIR / "movies.dat"
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1].strip()
            genres = parts[2].strip().split("|")

            # Extract year from title
            match = year_re.search(title)
            year = int(match.group(1)) if match else None

            movies.append((movie_id, title, year, genres))
    logger.info(f"Parsed {len(movies):,} movies from movies.dat")
    return movies


def parse_ratings():
    """Parse ratings.dat → list of (user_id, movie_id, rating, timestamp)."""
    ratings = []
    path = EXTRACT_DIR / "ratings.dat"
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            ratings.append((
                int(parts[0]),      # user_id
                int(parts[1]),      # movie_id
                float(parts[2]),    # rating
                int(parts[3]),      # timestamp
            ))
    logger.info(f"Parsed {len(ratings):,} ratings from ratings.dat")
    return ratings


# ─── Compute aggregate movie stats ───────────────────────────────────────────
def compute_movie_stats(ratings, movies):
    """Compute avg_rating and rating_count per movie.

    Returns: dict[movie_id] → (avg_rating, rating_count)
    """
    from collections import defaultdict
    movie_ratings = defaultdict(list)
    for _, movie_id, rating, _ in ratings:
        movie_ratings[movie_id].append(rating)

    stats = {}
    for movie_id, _, _, _ in movies:
        r_list = movie_ratings.get(movie_id, [])
        if r_list:
            stats[movie_id] = (float(np.mean(r_list)), len(r_list))
        else:
            stats[movie_id] = (0.0, 0)
    return stats


# ─── Bulk insert ──────────────────────────────────────────────────────────────
def load_into_db(conn, users, movies, ratings, movie_stats, use_sqlite: bool):
    """Create tables and bulk-insert all data."""

    cur = conn.cursor()

    # Create schema
    logger.info("Creating tables ...")
    schema = SQLITE_SCHEMA if use_sqlite else POSTGRES_SCHEMA
    for statement in schema.split(";"):
        statement = statement.strip()
        if statement:
            cur.execute(statement)
    conn.commit()

    # Insert users
    logger.info("Inserting users ...")
    if use_sqlite:
        cur.executemany(
            "INSERT INTO users (user_id, gender, age, occupation, zip_code) VALUES (?, ?, ?, ?, ?)",
            users
        )
    else:
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            "INSERT INTO users (user_id, gender, age, occupation, zip_code) VALUES %s",
            users,
            page_size=5000,
        )
    conn.commit()
    logger.info(f"  Inserted {len(users):,} users")

    # Insert movies (with avg_rating and rating_count)
    logger.info("Inserting movies ...")
    movie_rows = []
    for movie_id, title, year, genres in movies:
        avg_r, count = movie_stats.get(movie_id, (0.0, 0))
        if use_sqlite:
            # SQLite: store genres as pipe-separated string
            movie_rows.append((movie_id, title, year, "|".join(genres), avg_r, count))
        else:
            # PostgreSQL: store as TEXT array
            movie_rows.append((movie_id, title, year, genres, avg_r, count))

    if use_sqlite:
        cur.executemany(
            "INSERT INTO movies (movie_id, title, year, genres, avg_rating, rating_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            movie_rows,
        )
    else:
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            "INSERT INTO movies (movie_id, title, year, genres, avg_rating, rating_count) VALUES %s",
            movie_rows,
            page_size=5000,
        )
    conn.commit()
    logger.info(f"  Inserted {len(movie_rows):,} movies")

    # Insert ratings (in batches)
    logger.info("Inserting ratings ...")
    batch_size = 50000
    for i in range(0, len(ratings), batch_size):
        batch = ratings[i : i + batch_size]
        if use_sqlite:
            cur.executemany(
                "INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
                batch,
            )
        else:
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                "INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES %s",
                batch,
                page_size=5000,
            )
        conn.commit()
        logger.info(f"  Inserted batch {i // batch_size + 1}: {len(batch):,} ratings")

    logger.info(f"  Total ratings inserted: {len(ratings):,}")


# ─── Verify ──────────────────────────────────────────────────────────────────
def verify(conn, use_sqlite: bool):
    """Verify row counts."""
    cur = conn.cursor()
    for table in ["users", "movies", "ratings", "feedback_events"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        logger.info(f"  {table}: {count:,} rows")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("MovieLens 1M Data Loading Script")
    logger.info("=" * 60)

    # Step 1: Download & extract
    download_and_extract()

    # Step 2: Parse raw files
    users = parse_users()
    movies = parse_movies()
    ratings = parse_ratings()

    # Step 3: Compute movie stats
    movie_stats = compute_movie_stats(ratings, movies)

    # Step 4: Connect and load
    conn = get_sync_connection()
    use_sqlite = is_sqlite(conn)
    backend = "SQLite" if use_sqlite else "PostgreSQL"
    logger.info(f"Using database backend: {backend}")

    try:
        load_into_db(conn, users, movies, ratings, movie_stats, use_sqlite)

        # Step 5: Verify
        logger.info("Verifying row counts ...")
        verify(conn, use_sqlite)
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
