#!/usr/bin/env python3
"""
MovieLens 1M Data Loading Script
=================================
Downloads ml-1m.zip, parses .dat files, creates PostgreSQL tables, and
bulk-inserts all data.

Run from the project root:
    python scripts/load_data.py
"""

import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv
from loguru import logger

# Ensure project root is on sys.path so we can import api.database
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / "ml-1m.zip"
EXTRACT_DIR = DATA_DIR / "ml-1m"

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

# ---------------------------------------------------------------------------
# SQL Schema (exactly per spec section 4.1)
# ---------------------------------------------------------------------------
DROP_TABLES_SQL = """
DROP TABLE IF EXISTS feedback_events CASCADE;
DROP TABLE IF EXISTS ratings CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS users CASCADE;
"""

CREATE_USERS_SQL = """
CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    gender      CHAR(1),
    age         INTEGER,
    occupation  INTEGER,
    zip_code    VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

CREATE_MOVIES_SQL = """
CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    year        INTEGER,
    genres      TEXT[],
    avg_rating  FLOAT,
    rating_count INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

CREATE_RATINGS_SQL = """
CREATE TABLE ratings (
    rating_id   SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      FLOAT NOT NULL,
    timestamp   BIGINT,
    implicit    BOOLEAN GENERATED ALWAYS AS (rating >= 4.0) STORED,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ratings_user     ON ratings(user_id);
CREATE INDEX idx_ratings_movie    ON ratings(movie_id);
CREATE INDEX idx_ratings_implicit ON ratings(user_id, implicit);
"""

CREATE_FEEDBACK_SQL = """
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


# ---------------------------------------------------------------------------
# Download & Extract
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    """Download ml-1m.zip if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        logger.info(f"Dataset zip already exists at {ZIP_PATH}")
        return

    logger.info(f"Downloading MovieLens 1M from {MOVIELENS_URL} ...")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
        print(f"\r  Progress: {pct:.1f}% ({downloaded:,} / {total_size:,} bytes)", end="", flush=True)

    urlretrieve(MOVIELENS_URL, str(ZIP_PATH), reporthook=_progress)
    print()  # newline after progress bar
    logger.info("Download complete.")


def extract_dataset() -> None:
    """Unzip ml-1m.zip into data/raw/ml-1m/."""
    if EXTRACT_DIR.exists() and (EXTRACT_DIR / "users.dat").exists():
        logger.info(f"Data already extracted at {EXTRACT_DIR}")
        return

    logger.info(f"Extracting {ZIP_PATH} → {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(str(DATA_DIR))
    logger.info("Extraction complete.")


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------
def parse_users(filepath: Path) -> list[tuple]:
    """Parse users.dat → list of (user_id, gender, age, occupation, zip_code)."""
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 5:
                continue
            user_id = int(parts[0])
            gender = parts[1]
            age = int(parts[2])
            occupation = int(parts[3])
            zip_code = parts[4]
            rows.append((user_id, gender, age, occupation, zip_code))
    return rows


def parse_movies(filepath: Path) -> list[tuple]:
    """Parse movies.dat → list of (movie_id, title, year, genres_list).

    Year is extracted from the title with regex ``\\((\\d{4})\\)``.
    Genres are split on ``|`` into a Python list.
    """
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id = int(parts[0])
            title = parts[1].strip()
            genres_str = parts[2].strip()

            # Extract year from title
            m = YEAR_RE.search(title)
            year = int(m.group(1)) if m else None

            # Split genres
            genres = genres_str.split("|") if genres_str else []

            rows.append((movie_id, title, year, genres))
    return rows


def parse_ratings(filepath: Path) -> list[tuple]:
    """Parse ratings.dat → list of (user_id, movie_id, rating, timestamp)."""
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])
            rows.append((user_id, movie_id, rating, timestamp))
    return rows


# ---------------------------------------------------------------------------
# Compute movie aggregate stats
# ---------------------------------------------------------------------------
def compute_movie_stats(ratings_rows: list[tuple]) -> dict[int, tuple[float, int]]:
    """Compute avg_rating and rating_count per movie_id.

    Returns {movie_id: (avg_rating, rating_count)}.
    """
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for _, movie_id, rating, _ in ratings_rows:
        sums[movie_id] += rating
        counts[movie_id] += 1
    return {
        mid: (sums[mid] / counts[mid], counts[mid])
        for mid in counts
    }


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def create_tables(conn) -> None:
    """Drop existing tables and recreate them from scratch."""
    with conn.cursor() as cur:
        logger.info("Dropping existing tables (if any) ...")
        cur.execute(DROP_TABLES_SQL)

        logger.info("Creating users table ...")
        cur.execute(CREATE_USERS_SQL)

        logger.info("Creating movies table ...")
        cur.execute(CREATE_MOVIES_SQL)

        logger.info("Creating ratings table ...")
        cur.execute(CREATE_RATINGS_SQL)

        logger.info("Creating feedback_events table ...")
        cur.execute(CREATE_FEEDBACK_SQL)

    conn.commit()
    logger.info("All tables created successfully.")


def bulk_insert_users(conn, users: list[tuple]) -> None:
    """Bulk-insert user rows using execute_values."""
    from psycopg2.extras import execute_values

    sql = "INSERT INTO users (user_id, gender, age, occupation, zip_code) VALUES %s"
    with conn.cursor() as cur:
        execute_values(cur, sql, users, page_size=2000)
    conn.commit()
    logger.info(f"Inserted {len(users):,} users.")


def bulk_insert_movies(conn, movies: list[tuple], stats: dict[int, tuple[float, int]]) -> None:
    """Bulk-insert movie rows (with avg_rating and rating_count) using execute_values."""
    from psycopg2.extras import execute_values

    # Build rows with stats
    rows = []
    for movie_id, title, year, genres in movies:
        avg_rating, rating_count = stats.get(movie_id, (None, 0))
        rows.append((movie_id, title, year, genres, avg_rating, rating_count))

    sql = "INSERT INTO movies (movie_id, title, year, genres, avg_rating, rating_count) VALUES %s"

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="(%s, %s, %s, %s::TEXT[], %s, %s)",
            page_size=2000,
        )
    conn.commit()
    logger.info(f"Inserted {len(rows):,} movies.")


def bulk_insert_ratings(conn, ratings: list[tuple]) -> None:
    """Bulk-insert rating rows using execute_values (batched for large datasets)."""
    from psycopg2.extras import execute_values

    sql = "INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES %s"
    batch_size = 50_000

    with conn.cursor() as cur:
        for i in range(0, len(ratings), batch_size):
            batch = ratings[i : i + batch_size]
            execute_values(cur, sql, batch, page_size=5000)
            logger.debug(f"  Inserted ratings batch {i:,} – {i + len(batch):,}")
    conn.commit()
    logger.info(f"Inserted {len(ratings):,} ratings.")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_counts(conn) -> None:
    """Log row counts for all tables."""
    with conn.cursor() as cur:
        for table in ("users", "movies", "ratings", "feedback_events"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            logger.info(f"  {table}: {count:,} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("MovieLens 1M Data Loading Script")
    logger.info("=" * 60)

    # 1. Download & extract
    download_dataset()
    extract_dataset()

    # 2. Parse data files
    logger.info("Parsing users.dat ...")
    users = parse_users(EXTRACT_DIR / "users.dat")
    logger.info(f"  Parsed {len(users):,} users.")

    logger.info("Parsing movies.dat ...")
    movies = parse_movies(EXTRACT_DIR / "movies.dat")
    logger.info(f"  Parsed {len(movies):,} movies.")

    logger.info("Parsing ratings.dat ...")
    ratings = parse_ratings(EXTRACT_DIR / "ratings.dat")
    logger.info(f"  Parsed {len(ratings):,} ratings.")

    # 3. Compute movie stats
    logger.info("Computing per-movie avg_rating and rating_count ...")
    movie_stats = compute_movie_stats(ratings)
    logger.info(f"  Stats computed for {len(movie_stats):,} movies.")

    # 4. Connect to PostgreSQL and load
    from api.database import get_sync_connection

    logger.info("Connecting to PostgreSQL ...")
    conn = get_sync_connection()
    try:
        create_tables(conn)

        logger.info("Bulk inserting users ...")
        bulk_insert_users(conn, users)

        logger.info("Bulk inserting movies ...")
        bulk_insert_movies(conn, movies, movie_stats)

        logger.info("Bulk inserting ratings ...")
        bulk_insert_ratings(conn, ratings)

        logger.info("-" * 40)
        logger.info("Verifying row counts:")
        verify_counts(conn)
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("Data loading complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
