# 🎬 Production-Grade Recommender System — Full Architecture Spec

> **Purpose:** This document is a complete build specification for an AI coding assistant.
> It covers every component, model, data flow, API contract, and deployment detail needed to
> scaffold a production-style recommender system using **100% free-tier infrastructure**.
> The domain is **movie recommendations** using the publicly available **MovieLens 1M dataset**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack (Free Tier Only)](#3-tech-stack-free-tier-only)
4. [Data Layer](#4-data-layer)
5. [ML Pipeline — Stage 1: Candidate Generation (Two-Tower Model)](#5-ml-pipeline--stage-1-candidate-generation-two-tower-model)
6. [ML Pipeline — Stage 2: Ranking Model](#6-ml-pipeline--stage-2-ranking-model)
7. [ML Pipeline — Stage 3: Re-ranking & Business Rules](#7-ml-pipeline--stage-3-re-ranking--business-rules)
8. [Feature Engineering](#8-feature-engineering)
9. [API Design (FastAPI)](#9-api-design-fastapi)
10. [Caching Strategy](#10-caching-strategy)
11. [Frontend (Streamlit)](#11-frontend-streamlit)
12. [Training & Evaluation Pipeline](#12-training--evaluation-pipeline)
13. [Project Directory Structure](#13-project-directory-structure)
14. [Environment Variables & Configuration](#14-environment-variables--configuration)
15. [Deployment (Render + Hugging Face)](#15-deployment-render--hugging-face)
16. [Data Flow — End to End](#16-data-flow--end-to-end)
17. [Evaluation Metrics](#17-evaluation-metrics)
18. [Key Design Decisions & Rationale](#18-key-design-decisions--rationale)

---

## 1. Project Overview

### What is being built

A multi-stage recommender system that mimics how production systems at YouTube, Netflix,
and Amazon operate. Given a `user_id`, the system returns a ranked list of top-N movie
recommendations in under 200ms.

### The three-stage funnel (industry standard)

```
Entire Item Catalog (~3,700 movies)
        │
        ▼  Stage 1: Candidate Generation (Two-Tower Neural Net + ANN Search)
 ~200 candidates
        │
        ▼  Stage 2: Ranking (LightGBM + feature crosses)
  ~20 candidates
        │
        ▼  Stage 3: Re-ranking (diversity, freshness, business rules)
   Top-10 results returned to user
```

### Domain & Dataset

- **Dataset:** MovieLens 1M (`ml-1m`)
  - 1,000,209 ratings from 6,040 users across 3,706 movies
  - User features: age, gender, occupation, zip code
  - Item features: title, genres (multi-label)
  - Ratings: 1–5 stars (treated as implicit feedback: ≥4 = positive signal)
- **Download:** https://grouplens.org/datasets/movielens/1m/
- **License:** Free for non-commercial research/education use

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
│         Streamlit UI  ──────────────────▶  FastAPI Backend          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTP REST
┌───────────────────────────────▼─────────────────────────────────────┐
│                     API LAYER  (FastAPI)                            │
│  /recommend/{user_id}   /similar/{item_id}   /health   /feedback    │
└──────┬────────────────────────┬────────────────────────┬────────────┘
       │                        │                        │
       ▼                        ▼                        ▼
┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  Redis Cache │    │  Recommendation   │    │  Feedback Ingestion  │
│  (Upstash)   │    │  Engine           │    │  (writes to DB)      │
│  TTL=1h      │    └────────┬──────────┘    └──────────────────────┘
└──────────────┘             │
                             │ 3-stage pipeline
                  ┌──────────▼──────────────────┐
                  │  Stage 1: Candidate Gen      │
                  │  Two-Tower Net → FAISS ANN   │
                  │  Returns top-200 candidates  │
                  └──────────┬──────────────────┘
                             │
                  ┌──────────▼──────────────────┐
                  │  Stage 2: Ranking            │
                  │  LightGBM scorer             │
                  │  Returns top-20 with scores  │
                  └──────────┬──────────────────┘
                             │
                  ┌──────────▼──────────────────┐
                  │  Stage 3: Re-ranking         │
                  │  Diversity filter (MMR)      │
                  │  Already-watched filter      │
                  │  Returns final top-10        │
                  └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                      │
│   PostgreSQL (Supabase free)           FAISS Index (in-memory)     │
│   ├── users table                      Item embeddings (512-dim)   │
│   ├── movies table                                                  │
│   ├── ratings table                    Model Registry              │
│   └── feedback_events table            ├── two_tower.pt            │
│                                        ├── lgbm_ranker.pkl         │
│                                        └── item_embeddings.npy     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack (Free Tier Only)

| Layer | Technology | Free Tier Details |
|---|---|---|
| **API Backend** | FastAPI (Python 3.11) | Self-hosted on Render |
| **ML Framework** | PyTorch 2.x | CPU inference only |
| **Gradient Boosting** | LightGBM | In-process, no extra cost |
| **ANN Search** | FAISS (`faiss-cpu`) | In-memory, rebuilt on startup |
| **Primary Database** | PostgreSQL via **Supabase** | 500MB storage, free forever |
| **Cache** | Redis via **Upstash** | 10,000 req/day, 256MB free |
| **Model Storage** | **Hugging Face Hub** (private repo) | Free for models < 10GB |
| **API Hosting** | **Render** (free web service) | 750 hrs/month, spins down |
| **Frontend** | **Streamlit Community Cloud** | Free, connects to FastAPI |
| **Experiment Tracking** | **MLflow** on local / Kaggle | Artifacts logged locally |
| **Data Processing** | Pandas, NumPy, scikit-learn | All free |
| **CI/CD** | GitHub Actions | 2,000 min/month free |

> **Note on Render free tier:** Web services spin down after 15 minutes of inactivity.
> The first cold-start request will take ~60 seconds (FAISS index rebuild + model load).
> All subsequent requests are fast. This is acceptable for a portfolio/internship project.

---

## 4. Data Layer

### 4.1 PostgreSQL Schema (Supabase)

```sql
-- users.sql
CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    gender      CHAR(1),          -- 'M' or 'F'
    age         INTEGER,          -- bucketed: 1,18,25,35,45,50,56
    occupation  INTEGER,          -- 0-20 per MovieLens mapping
    zip_code    VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- movies.sql
CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    year        INTEGER,          -- extracted from title
    genres      TEXT[],           -- array: ['Action','Comedy',...]
    avg_rating  FLOAT,
    rating_count INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- ratings.sql  (core interaction matrix)
CREATE TABLE ratings (
    rating_id   SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      FLOAT NOT NULL,   -- 1.0 to 5.0
    timestamp   BIGINT,
    implicit    BOOLEAN GENERATED ALWAYS AS (rating >= 4.0) STORED,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ratings_user    ON ratings(user_id);
CREATE INDEX idx_ratings_movie   ON ratings(movie_id);
CREATE INDEX idx_ratings_implicit ON ratings(user_id, implicit);

-- feedback_events.sql  (online feedback from API usage)
CREATE TABLE feedback_events (
    event_id    SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(user_id),
    movie_id    INTEGER REFERENCES movies(movie_id),
    event_type  VARCHAR(20),      -- 'click', 'watch', 'skip', 'rate'
    rating      FLOAT,            -- nullable, only for 'rate' events
    session_id  VARCHAR(64),
    timestamp   TIMESTAMP DEFAULT NOW()
);
```

### 4.2 Data Loading Script

File: `scripts/load_data.py`

This script must:
1. Download and unzip `ml-1m.zip` from the GroupLens URL
2. Parse `users.dat`, `movies.dat`, `ratings.dat` (`::`-separated)
3. Extract year from movie title using regex `\((\d{4})\)$`
4. Split genres string `"Action|Comedy"` into a PostgreSQL TEXT array
5. Compute `avg_rating` and `rating_count` per movie and persist
6. Bulk-insert using `psycopg2.extras.execute_values` for performance
7. Log row counts at completion

---

## 5. ML Pipeline — Stage 1: Candidate Generation (Two-Tower Model)

### 5.1 Architecture Overview

The Two-Tower model learns separate embedding spaces for users and items so that
**dot-product similarity** in that shared space approximates relevance. At inference,
user embedding is computed once, and FAISS returns the top-K nearest item embeddings
in sub-millisecond time.

```
USER TOWER                           ITEM TOWER
──────────────────────               ──────────────────────
user_id embedding (64-dim)           item_id embedding (64-dim)
age embedding (16-dim)               genre multi-hot → Linear(18→32)
gender embedding (4-dim)             year_norm → Linear(1→8)
occupation embedding (16-dim)        avg_rating_norm (1-dim)
watched_genres (18-dim avg)          rating_count_log (1-dim)
         │                                     │
    Concat → 118-dim                      Concat → 106-dim
         │                                     │
   Linear(118→256)                       Linear(106→256)
   BatchNorm + ReLU                      BatchNorm + ReLU
   Dropout(0.2)                          Dropout(0.2)
   Linear(256→128)                       Linear(256→128)
   BatchNorm + ReLU                      BatchNorm + ReLU
   Linear(128→64)                        Linear(128→64)
   L2 Normalize                          L2 Normalize
         │                                     │
         └──────────────┬──────────────────────┘
                        │
              cosine_similarity(u, i)
                        │
              BPR Loss (Bayesian Personalized Ranking)
```

### 5.2 Training Objective — BPR Loss

Use **Bayesian Personalized Ranking (BPR)**: for each user, sample a positive item
(rated ≥ 4) and a negative item (not interacted with). The model learns:
`score(user, pos) > score(user, neg)`.

```python
# BPR loss formula
loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
```

### 5.3 Model Class: `TwoTowerModel`

File: `models/two_tower.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, n_users, n_occupations=21, n_genres=18, embed_dim=64):
        super().__init__()
        self.user_emb      = nn.Embedding(n_users + 1, embed_dim)
        self.age_emb       = nn.Embedding(8, 16)       # 7 age buckets + 1 pad
        self.gender_emb    = nn.Embedding(3, 4)        # M/F + pad
        self.occ_emb       = nn.Embedding(n_occupations + 1, 16)
        self.genre_proj    = nn.Linear(n_genres, 18)

        input_dim = embed_dim + 16 + 4 + 16 + 18
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, user_id, age_bucket, gender, occupation, watched_genre_vec):
        x = torch.cat([
            self.user_emb(user_id),
            self.age_emb(age_bucket),
            self.gender_emb(gender),
            self.occ_emb(occupation),
            self.genre_proj(watched_genre_vec)
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items, n_genres=18, embed_dim=64):
        super().__init__()
        self.item_emb   = nn.Embedding(n_items + 1, embed_dim)
        self.genre_proj = nn.Linear(n_genres, 32)
        self.year_proj  = nn.Linear(1, 8)

        input_dim = embed_dim + 32 + 8 + 2  # +2 for avg_rating, log_count
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, item_id, genre_vec, year_norm, avg_rating_norm, log_count):
        x = torch.cat([
            self.item_emb(item_id),
            self.genre_proj(genre_vec),
            self.year_proj(year_norm.unsqueeze(-1)),
            avg_rating_norm.unsqueeze(-1),
            log_count.unsqueeze(-1)
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_tower = UserTower(n_users)
        self.item_tower = ItemTower(n_items)

    def forward(self, user_batch, item_batch):
        u_emb = self.user_tower(**user_batch)
        i_emb = self.item_tower(**item_batch)
        return (u_emb * i_emb).sum(dim=-1)    # dot product → scalar score
```

### 5.4 Dataset Class: `MovieLensDataset`

File: `training/dataset.py`

Each sample is a triplet `(user_features, positive_item_features, negative_item_features)`.
Negative items are sampled uniformly from movies the user has NOT rated.
Use negative sampling ratio of **4 negatives per positive**.

### 5.5 Training Config

File: `training/train_two_tower.py`

```python
CONFIG = {
    "batch_size":     2048,
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "epochs":         30,
    "optimizer":      "AdamW",
    "scheduler":      "CosineAnnealingLR",
    "embed_dim":      64,
    "neg_sample_ratio": 4,
    "val_split":      0.1,    # time-based split (last 10% of timestamps = val)
    "patience":       5,      # early stopping on val Recall@20
    "model_save_path": "artifacts/two_tower.pt",
}
```

### 5.6 FAISS Index (ANN Search)

File: `retrieval/faiss_index.py`

```python
import faiss
import numpy as np

class FAISSItemIndex:
    """
    Stores all item embeddings in a FAISS flat inner-product index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """
    def __init__(self, dim: int = 64):
        self.index = faiss.IndexFlatIP(dim)   # IP = inner product
        self.item_ids: list[int] = []

    def build(self, item_embeddings: np.ndarray, item_ids: list[int]):
        """item_embeddings: shape (N, dim), float32, L2-normalized"""
        self.item_ids = item_ids
        self.index.add(item_embeddings.astype(np.float32))

    def search(self, user_embedding: np.ndarray, top_k: int = 200) -> list[int]:
        """Returns top_k item_ids sorted by descending cosine similarity."""
        scores, indices = self.index.search(
            user_embedding.reshape(1, -1).astype(np.float32), top_k
        )
        return [self.item_ids[i] for i in indices[0]]
```

At API startup, pre-compute ALL item embeddings once and load into FAISS.
Store the resulting `item_embeddings.npy` as an artifact alongside the model checkpoint.

---

## 6. ML Pipeline — Stage 2: Ranking Model

### 6.1 Purpose

Take the ~200 candidates from Stage 1 and produce a precise relevance score per
`(user, item)` pair using richer features that would be too expensive to compute
over the full catalog.

### 6.2 Model: LightGBM Ranker (LambdaRank objective)

File: `models/ranker.py`

```python
import lightgbm as lgb

LGBM_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     [5, 10, 20],
    "num_leaves":       63,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":        0.1,
    "n_jobs":           -1,
    "verbose":          -1,
}
```

The `lambdarank` objective directly optimizes NDCG, making it ideal for ranking tasks.

### 6.3 Ranking Feature Vector (per user-item pair)

Each training row is one `(user, candidate_movie)` pair. Label = 1 if rating ≥ 4, 0 otherwise.

```
USER FEATURES (static)
  user_age_norm          float  — age normalized 0-1
  user_gender_enc        int    — 0=F, 1=M
  user_occupation        int    — 0-20
  user_avg_rating_given  float  — mean rating this user gives
  user_rating_count      int    — total ratings by user
  user_top_genres        float[18] — genre preference vector

ITEM FEATURES (static)
  item_avg_rating        float
  item_rating_count_log  float
  item_year_norm         float
  item_genres            float[18] — multi-hot genre vector

INTERACTION FEATURES (key for ranking quality)
  two_tower_score        float  — cosine similarity from Stage 1
  genre_overlap          float  — dot(user_top_genres, item_genres)
  user_has_rated_genre   int    — 1 if user rated ≥3 movies in this genre
  rating_gap_years       float  — years since user first rated similar genre
  popularity_percentile  float  — item's rating_count percentile (0-1)
  is_sequel              int    — heuristic: title contains digit or "II/III"
```

Total feature vector: ~45 features.

### 6.4 Training the Ranker

File: `training/train_ranker.py`

1. Load all positive ratings (≥ 4.0) as label=1 samples.
2. For each user, sample 10 negative items (not rated or rated < 3) as label=0.
3. Compute the full feature vector for each row.
4. Use time-based split: train on ratings before timestamp T, val on after.
5. Use `lgb.Dataset` with `group` parameter (required for lambdarank): group = count of rows per user.
6. Save model with `model.save_model("artifacts/lgbm_ranker.txt")`.

---

## 7. ML Pipeline — Stage 3: Re-ranking & Business Rules

File: `pipeline/reranker.py`

Takes the top-20 from Stage 2 and applies:

### 7.1 Already-Watched Filter

Remove any movie the user has already rated (queried from `ratings` table).

### 7.2 Maximal Marginal Relevance (MMR) for Diversity

MMR balances relevance and diversity by iteratively selecting items that are
both highly scored AND dissimilar from already-selected items.

```python
def mmr_rerank(
    candidates: list[dict],      # [{"movie_id": ..., "score": ..., "genre_vec": ...}]
    top_n: int = 10,
    lambda_mmr: float = 0.7      # 0.7 = favor relevance, 0.3 = favor diversity
) -> list[dict]:
    """
    Iteratively pick the item maximizing:
        λ * relevance_score - (1-λ) * max_similarity_to_selected
    genre_vec similarity is cosine similarity between 18-dim genre vectors.
    """
    selected = []
    remaining = candidates.copy()
    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x["score"])
        else:
            def mmr_score(item):
                rel = item["score"]
                max_sim = max(cosine_sim(item["genre_vec"], s["genre_vec"]) for s in selected)
                return lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected
```

### 7.3 Final Output Shape

```json
[
  {
    "movie_id": 1196,
    "title": "Star Wars: Episode V - The Empire Strikes Back (1980)",
    "genres": ["Action", "Adventure", "Sci-Fi"],
    "two_tower_score": 0.92,
    "ranker_score": 0.87,
    "final_rank": 1
  },
  ...
]
```

---

## 8. Feature Engineering

File: `features/feature_store.py`

All feature computation must be cached to avoid recomputing at inference time.

### 8.1 Precomputed Features (computed once during data loading)

| Feature Name | Type | Description |
|---|---|---|
| `user_genre_pref_vec` | float[18] | Avg genre vector of positively-rated movies |
| `user_avg_rating` | float | Mean of all ratings given |
| `user_rating_count` | int | Total number of ratings |
| `item_genre_vec` | float[18] | Multi-hot genre encoding |
| `item_year_norm` | float | (year - 1920) / 100 |
| `item_avg_rating` | float | Global average rating |
| `item_rating_count_log` | float | `log1p(rating_count)` |
| `item_popularity_pct` | float | Percentile rank by count |

### 8.2 On-the-Fly Features (computed per request)

| Feature Name | How to compute |
|---|---|
| `genre_overlap` | `dot(user_genre_pref_vec, item_genre_vec)` |
| `two_tower_score` | From Stage 1 output |
| `user_has_rated_genre` | Check user's rating history |

### 8.3 Feature Serialization

- Save all precomputed features as a single `features.pkl` (dictionary keyed by entity ID)
- Load into memory at API startup
- Total memory footprint: ~50MB for MovieLens 1M

---

## 9. API Design (FastAPI)

File: `api/main.py`

### 9.1 Application Setup

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routers import recommend, health, feedback, similar

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load all models into memory
    from api.model_registry import ModelRegistry
    app.state.registry = ModelRegistry()
    app.state.registry.load_all()       # loads two_tower.pt, lgbm_ranker.pkl,
                                        # item_embeddings.npy, FAISS index, features.pkl
    yield
    # Shutdown: nothing to clean up

app = FastAPI(
    title="MovieRec API",
    description="Multi-stage recommender system",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(recommend.router, prefix="/api/v1")
app.include_router(similar.router,   prefix="/api/v1")
app.include_router(feedback.router,  prefix="/api/v1")
app.include_router(health.router)
```

### 9.2 Model Registry

File: `api/model_registry.py`

```python
class ModelRegistry:
    two_tower:      TwoTowerModel
    ranker:         lgb.Booster
    faiss_index:    FAISSItemIndex
    features:       dict           # {"user": {...}, "item": {...}}
    item_meta:      pd.DataFrame   # movie_id, title, genres, ...

    def load_all(self):
        """Load all artifacts from ./artifacts/ directory."""
        # Download from HuggingFace Hub if artifacts/ is empty (first cold start)
        ...
```

### 9.3 Endpoints

#### `GET /api/v1/recommend/{user_id}`

Returns top-N movie recommendations for a user.

**Query params:**
- `top_n: int = 10` — number of results (max 50)
- `exclude_watched: bool = True` — filter already-seen movies
- `diversity: float = 0.7` — MMR lambda (0=pure diversity, 1=pure relevance)

**Response (200 OK):**
```json
{
  "user_id": 42,
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 1196,
      "title": "The Empire Strikes Back (1980)",
      "genres": ["Action", "Sci-Fi"],
      "score": 0.921,
      "explanation": "Because you liked Star Wars (1977)"
    }
  ],
  "latency_ms": 47,
  "cache_hit": false
}
```

**Response (404 Not Found):**
```json
{"detail": "User 999999 not found in the system."}
```

#### `GET /api/v1/similar/{movie_id}`

Returns top-N movies similar to a given movie (item-to-item).
Uses cosine similarity on item embeddings from the FAISS index.

**Query params:** `top_n: int = 10`

#### `POST /api/v1/feedback`

Ingest a user interaction event (for future model retraining).

**Request body:**
```json
{
  "user_id": 42,
  "movie_id": 1196,
  "event_type": "click",
  "session_id": "abc123"
}
```

#### `GET /health`

Returns `{"status": "ok", "models_loaded": true, "faiss_index_size": 3706}`

### 9.4 Recommendation Pipeline (inside the endpoint)

```python
async def get_recommendations(user_id: int, top_n: int, ...):
    # 1. Check Redis cache
    cache_key = f"rec:{user_id}:{top_n}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. Build user feature tensor
    user_features = feature_store.get_user_features(user_id)

    # 3. Stage 1: Get user embedding → FAISS ANN → top-200 candidates
    user_emb = two_tower.user_tower(**user_features)
    candidate_ids = faiss_index.search(user_emb.numpy(), top_k=200)

    # 4. Stage 2: Rank with LightGBM
    rank_features = feature_store.build_rank_features(user_id, candidate_ids)
    scores = ranker.predict(rank_features)
    top20_ids = candidate_ids[np.argsort(scores)[::-1][:20]]

    # 5. Stage 3: Re-rank (filter watched + MMR diversity)
    watched = db.get_watched_movie_ids(user_id)
    top20_ids = [m for m in top20_ids if m not in watched]
    final = mmr_rerank(top20_ids, top_n=top_n)

    # 6. Cache result for 1 hour
    await redis.setex(cache_key, 3600, json.dumps(final))
    return final
```

---

## 10. Caching Strategy

File: `api/cache.py`

Use **Upstash Redis** (free tier: 10,000 commands/day, 256MB).

| Cache Key Pattern | TTL | Content |
|---|---|---|
| `rec:{user_id}:{top_n}` | 1 hour | Full recommendation list |
| `sim:{movie_id}:{top_n}` | 24 hours | Similar items list |
| `user_feats:{user_id}` | 12 hours | Serialized user feature dict |
| `item_meta:{movie_id}` | 24 hours | Movie metadata JSON |

**Cache invalidation:** When a feedback event arrives for `user_id`, delete `rec:{user_id}:*`
to force fresh recommendation on next request.

```python
import upstash_redis

redis = upstash_redis.Redis(
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)
```

---

## 11. Frontend (Streamlit)

File: `frontend/app.py`

### 11.1 Pages

**Page 1: User Recommendations**
- Sidebar: user_id input (1–6040), top_n slider (5–20), diversity slider
- Main: Grid of movie cards with poster placeholder, title, genres, score badge
- Show `⚡ Cache hit` or `🔄 Fresh` indicator
- Show total latency

**Page 2: Movie Explorer**
- Search box for movie title (calls `/api/v1/similar/{movie_id}`)
- Grid of similar movies

**Page 3: System Stats**
- Calls `/health` endpoint
- Shows model metadata, FAISS index size, cache hit rate (if tracked)

### 11.2 API Client

```python
import httpx
import streamlit as st

API_BASE = os.getenv("API_URL", "http://localhost:8000")

@st.cache_data(ttl=3600)
def fetch_recommendations(user_id: int, top_n: int) -> dict:
    r = httpx.get(f"{API_BASE}/api/v1/recommend/{user_id}", params={"top_n": top_n})
    r.raise_for_status()
    return r.json()
```

---

## 12. Training & Evaluation Pipeline

### 12.1 Notebook Structure (Kaggle / local Jupyter)

```
notebooks/
├── 01_data_exploration.ipynb       # EDA on MovieLens 1M
├── 02_feature_engineering.ipynb    # Build & save features.pkl
├── 03_train_two_tower.ipynb        # Train + eval Two-Tower
├── 04_train_ranker.ipynb           # Train + eval LightGBM ranker
├── 05_end_to_end_eval.ipynb        # Full pipeline offline evaluation
├── 06_ablation_study.ipynb         # Compare: CF-only vs Two-Tower vs Two-Tower+Ranker
```

### 12.2 Time-Based Train/Test Split

**DO NOT use random splits** — they leak temporal information.

```python
# Split on timestamp: train = first 90% of ratings (by time), test = last 10%
ratings_sorted = ratings_df.sort_values("timestamp")
split_idx = int(len(ratings_sorted) * 0.9)
train_df = ratings_sorted.iloc[:split_idx]
test_df  = ratings_sorted.iloc[split_idx:]

# For evaluation: use only users who appear in BOTH train and test
test_users = set(test_df["user_id"]) & set(train_df["user_id"])
```

### 12.3 Evaluation Protocol (Leave-Last-Out)

For each test user:
1. Treat their last-K positive interactions as ground truth
2. Use the pipeline to generate top-N recommendations
3. Compute metrics against ground truth

---

## 13. Project Directory Structure

```
recommender-system/
│
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── render.yaml                      # Render deployment config
│
├── data/
│   └── raw/                         # ml-1m/ unzipped here (git-ignored)
│
├── artifacts/                       # Model artifacts (git-ignored, loaded from HF Hub)
│   ├── two_tower.pt
│   ├── lgbm_ranker.txt
│   ├── item_embeddings.npy
│   └── features.pkl
│
├── scripts/
│   ├── load_data.py                 # Loads MovieLens into PostgreSQL
│   ├── precompute_features.py       # Builds features.pkl
│   └── precompute_embeddings.py     # Runs item tower over all items → .npy
│
├── models/
│   ├── two_tower.py                 # TwoTowerModel, UserTower, ItemTower
│   └── ranker.py                    # LightGBM config and helpers
│
├── training/
│   ├── dataset.py                   # MovieLensDataset (triplet sampling)
│   ├── train_two_tower.py           # Training loop + early stopping
│   └── train_ranker.py              # LightGBM training + feature builder
│
├── retrieval/
│   └── faiss_index.py               # FAISSItemIndex class
│
├── features/
│   └── feature_store.py             # FeatureStore class (precomputed + on-the-fly)
│
├── pipeline/
│   ├── candidate_generator.py       # Stage 1: wraps Two-Tower + FAISS
│   ├── ranker_pipeline.py           # Stage 2: wraps LightGBM + feature builder
│   └── reranker.py                  # Stage 3: MMR + watched filter
│
├── api/
│   ├── main.py                      # FastAPI app + lifespan
│   ├── model_registry.py            # ModelRegistry (loads all artifacts)
│   ├── cache.py                     # Upstash Redis client + helpers
│   ├── database.py                  # SQLAlchemy async engine + session
│   └── routers/
│       ├── recommend.py
│       ├── similar.py
│       ├── feedback.py
│       └── health.py
│
├── frontend/
│   └── app.py                       # Streamlit app
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_train_two_tower.ipynb
    ├── 04_train_ranker.ipynb
    ├── 05_end_to_end_eval.ipynb
    └── 06_ablation_study.ipynb
```

---

## 14. Environment Variables & Configuration

File: `.env.example`

```bash
# PostgreSQL (Supabase)
DATABASE_URL=postgresql+asyncpg://user:password@db.supabase.co:5432/postgres

# Redis (Upstash)
UPSTASH_REDIS_URL=https://xxxx.upstash.io
UPSTASH_REDIS_TOKEN=xxxx

# Hugging Face (for artifact download on cold start)
HF_TOKEN=hf_xxxxxxxxxxxx
HF_REPO_ID=your-username/movierec-artifacts

# App config
API_PORT=8000
ENV=production
LOG_LEVEL=INFO
TOP_N_DEFAULT=10
FAISS_TOP_K=200        # Candidates from Stage 1
RANKER_TOP_K=20        # Candidates passed to Stage 3
MMR_LAMBDA=0.7
CACHE_TTL_RECS=3600    # seconds
```

---

## 15. Deployment (Render + Hugging Face)

### 15.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download artifacts from HuggingFace Hub at build time
# (avoids cold-start penalty; update HF_REPO_ID with your repo)
RUN python scripts/download_artifacts.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 15.2 `render.yaml`

```yaml
services:
  - type: web
    name: movierec-api
    runtime: docker
    plan: free
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: UPSTASH_REDIS_URL
        sync: false
      - key: UPSTASH_REDIS_TOKEN
        sync: false
      - key: HF_TOKEN
        sync: false
      - key: HF_REPO_ID
        value: your-username/movierec-artifacts
    healthCheckPath: /health
```

### 15.3 Artifact Download Script

File: `scripts/download_artifacts.py`

```python
"""
Downloads model artifacts from HuggingFace Hub into ./artifacts/
Only runs if artifacts are missing (for Docker builds and cold starts).
"""
from huggingface_hub import hf_hub_download
import os

ARTIFACTS = [
    "two_tower.pt",
    "lgbm_ranker.txt",
    "item_embeddings.npy",
    "features.pkl",
]

def download_all():
    os.makedirs("artifacts", exist_ok=True)
    repo_id = os.environ["HF_REPO_ID"]
    token   = os.environ.get("HF_TOKEN")
    for fname in ARTIFACTS:
        dest = f"artifacts/{fname}"
        if not os.path.exists(dest):
            print(f"Downloading {fname}...")
            hf_hub_download(repo_id=repo_id, filename=fname, local_dir="artifacts", token=token)

if __name__ == "__main__":
    download_all()
```

### 15.4 Streamlit Deployment

Deploy `frontend/app.py` to **Streamlit Community Cloud** (https://streamlit.io/cloud).
Set `API_URL` in Streamlit's secrets management to point to your Render service URL.

---

## 16. Data Flow — End to End

```
1. OFFLINE (Kaggle / local machine)
   ─────────────────────────────────
   a. Download ml-1m.zip
   b. Run load_data.py → inserts into Supabase PostgreSQL
   c. Run precompute_features.py → saves features.pkl
   d. Run train_two_tower.py → saves two_tower.pt
   e. Run precompute_embeddings.py → saves item_embeddings.npy
   f. Run train_ranker.py → saves lgbm_ranker.txt
   g. Upload all 4 artifact files to HuggingFace Hub (private repo)

2. DEPLOYMENT
   ─────────────────────────────────
   a. Push code to GitHub
   b. Render auto-deploys Docker image
   c. On startup: download_artifacts.py pulls from HF Hub
   d. lifespan() loads all models into RAM + builds FAISS index
   e. Render health check hits /health → marks service UP

3. INFERENCE (per request)
   ─────────────────────────────────
   a. Streamlit sends GET /api/v1/recommend/42
   b. FastAPI checks Upstash Redis → cache miss
   c. user_tower(user_42_features) → 64-dim embedding
   d. FAISS.search(user_emb, k=200) → 200 candidate IDs (< 1ms)
   e. build_rank_features(user_42, candidates) → (200, 45) feature matrix
   f. lgbm.predict(features) → scores vector (< 5ms)
   g. Top-20 by score → filter watched → MMR rerank → top-10
   h. Write result to Redis (TTL=1h)
   i. Return JSON response

4. FEEDBACK LOOP
   ─────────────────────────────────
   a. POST /api/v1/feedback with click/watch events
   b. Insert into feedback_events table
   c. Invalidate Redis cache for that user
   d. (Weekly) retrain ranker with new feedback data
```

---

## 17. Evaluation Metrics

All metrics computed at **N=10** (Recall@10, NDCG@10, etc.)

| Metric | Formula | Target |
|---|---|---|
| **Recall@K** | `|relevant ∩ top_K| / |relevant|` | ≥ 0.25 |
| **Precision@K** | `|relevant ∩ top_K| / K` | ≥ 0.15 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | ≥ 0.30 |
| **MRR** | Mean Reciprocal Rank of first relevant item | ≥ 0.25 |
| **Coverage** | % of catalog appearing in top-10 across all users | ≥ 20% |
| **Diversity** | Avg pairwise genre dissimilarity within a list | ≥ 0.4 |
| **P99 Latency** | 99th percentile end-to-end response time | ≤ 200ms |

### Baseline Comparisons (Ablation Study)

| System | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|
| Random | ~0.01 | ~0.01 | Floor baseline |
| Popularity (most-rated) | ~0.08 | ~0.09 | Non-personalized |
| User-Based CF (cosine) | ~0.15 | ~0.17 | Classic approach |
| Matrix Factorization (ALS) | ~0.20 | ~0.22 | Standard ML |
| **Two-Tower (ours)** | ~0.24 | ~0.26 | Stage 1 only |
| **Two-Tower + LGBM** | ~0.28 | ~0.31 | Full pipeline |

---

## 18. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **BPR loss over BCE** | Implicit feedback — we know positives but not true negatives. BPR's pairwise objective handles this better than treating unobserved items as negatives. |
| **LightGBM for ranking (not a neural ranker)** | Faster to train, no GPU needed, interpretable feature importances, competitive NDCG on tabular data. Neural rankers (DCN, DeepFM) would need GPU for acceptable latency. |
| **FAISS `IndexFlatIP` over HNSW** | The catalog is small (3,706 items). Flat exact search is fast enough and avoids approximation errors. Use HNSW only if catalog > 100K items. |
| **In-memory FAISS (no Pinecone/Weaviate)** | Free tier databases have latency overhead. At 3,706 items × 64 dims × float32 = ~950KB. Fits trivially in RAM. |
| **Supabase over direct Postgres** | Supabase gives a free managed PostgreSQL with connection pooling (PgBouncer) and a REST API out of the box. No server to manage. |
| **Upstash over Redis Cloud** | Upstash has a serverless pricing model (per request) that is more suited to low-traffic portfolio projects. Free tier supports our use case. |
| **HF Hub for artifact storage** | Free, version-controlled model storage. `huggingface_hub` library makes download and upload trivially easy. |
| **L2 normalization before FAISS** | Converts inner product search into cosine similarity search, which is more robust to embedding magnitude variance across users. |
| **Time-based split** | Prevents data leakage: model only "knows" about interactions that would have been available at training time. Random splits are invalid for temporal recommendation tasks. |

---

## Requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
psycopg2-binary==2.9.9

torch==2.3.0
lightgbm==4.3.0
faiss-cpu==1.8.0
scikit-learn==1.5.0
pandas==2.2.0
numpy==1.26.4

upstash-redis==1.1.0
huggingface-hub==0.23.0

httpx==0.27.0
streamlit==1.35.0

python-dotenv==1.0.1
loguru==0.7.2
```

---

*End of specification. All components above must be implemented in full — no placeholder code.*
*Every class, function, and script referenced by name must exist as a real, working implementation.*
