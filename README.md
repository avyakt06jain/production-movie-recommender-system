# 🎬 Production-Grade Recommender System

A multi-stage recommender system that mimics production systems at YouTube, Netflix, and Amazon. Given a `user_id`, the system returns a ranked list of top-N movie recommendations in under 200ms.

## Architecture

```
Entire Item Catalog (~3,700 movies)
        │
        ▼  Stage 1: Candidate Generation (Two-Tower Neural Net + FAISS ANN)
 ~200 candidates
        │
        ▼  Stage 2: Ranking (LightGBM + feature crosses)
  ~20 candidates
        │
        ▼  Stage 3: Re-ranking (diversity, freshness, business rules)
   Top-10 results returned to user
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Backend | FastAPI (Python 3.11) |
| ML Framework | PyTorch 2.x (CPU) |
| Gradient Boosting | LightGBM |
| ANN Search | FAISS (faiss-cpu) |
| Database | PostgreSQL |
| Cache | Redis |
| Frontend | Streamlit |
| Model Storage | Hugging Face Hub |
| Deployment | Render (Docker) |

## Dataset

**MovieLens 1M** — 1,000,209 ratings from 6,040 users across 3,706 movies.

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Setup Database

Ensure PostgreSQL is running locally:

```bash
createdb movierec
```

### 3. Load Data

```bash
python scripts/load_data.py
```

### 4. Train Models

```bash
python scripts/precompute_features.py
python training/train_two_tower.py
python scripts/precompute_embeddings.py
python training/train_ranker.py
```

### 5. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start Frontend

```bash
streamlit run frontend/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/recommend/{user_id}` | Top-N movie recommendations |
| GET | `/api/v1/similar/{movie_id}` | Similar movies (item-to-item) |
| POST | `/api/v1/feedback` | Ingest user interaction events |
| GET | `/health` | Health check |

## Project Structure

```
recommender-system/
├── api/               # FastAPI backend
├── models/            # ML model definitions
├── training/          # Training scripts
├── pipeline/          # 3-stage inference pipeline
├── retrieval/         # FAISS index
├── features/          # Feature engineering
├── scripts/           # Data loading, artifact management
├── frontend/          # Streamlit UI
└── notebooks/         # Exploration & evaluation
```

## License

For educational/portfolio use. Dataset under MovieLens terms of use.
