# MovieRec: Machine Learning Recommender System

A machine learning recommendation system built to serve personalized movie suggestions. It leverages a **3-stage ranking funnel** (Candidate Generation → Ranking → Re-ranking) to prioritize high relevance and diversity.

---

## What is this?
This project is an end-to-end Machine Learning pipeline and web application trained on the **MovieLens 1M** dataset. It features an asynchronous **FastAPI backend** and a **Streamlit frontend**, integrating with PostgreSQL for data storage, Hugging Face for model loading, and Redis for caching.

### Key Features
- **Two-Tower Neural Network**: PyTorch-based model for generating 64-dimensional user and item embeddings.
- **Approximate Nearest Neighbors (ANN)**: Fast candidate generation over 3,700+ movies using **FAISS**.
- **LightGBM Ranker**: Tree-boosting algorithm (LambdaRank) evaluating dynamic features per user-item pair.
- **MMR Diversity Re-ranking**: Ensures recommendations maximize genre diversity.
- **Redis Caching**: Caches API responses to speed up load times.

---

## System Architecture

![System Architecture](imgs/recommender-system-diagram.png)

### Core Components

1. **Frontend (Streamlit)**: A clean, interactive web dashboard for browsing movies, adjusting preferences, and receiving personalized recommendations.
2. **API Layer (FastAPI)**: The backend that coordinates the ML funnel and handles database/cache communication.
3. **Caching (Redis)**: Caches API responses and item-item similarity vectors.
4. **Candidate Generation (Two-Tower & FAISS)**: Uses PyTorch embeddings to narrow down the catalog to the top 200 candidates via an Approximate Nearest Neighbor search.
5. **Scoring & Ranking (LightGBM)**: A LambdaRank model that computes a precise relevancy score for each candidate.
6. **Diversity Re-ranking (MMR)**: Applies Maximal Marginal Relevance to ensure the final top-10 list is diverse.
7. **Database (PostgreSQL)**: Stores metadata, user profiles, and ratings.
8. **Model Registry (Hugging Face Hub)**: The API downloads the trained models directly from the Hub on startup.

---

## Technical Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | PyTorch, LightGBM, FAISS, Scikit-Learn |
| **Backend & API** | Python 3.11, FastAPI, Uvicorn |
| **Database** | PostgreSQL, SQLAlchemy |
| **Caching** | Redis |
| **Frontend** | Streamlit |

---

## Performance & Stats
- **Data Scale**: Trained on 1,000,209 interactions, 6,040 users, and 3,706 movies.
- **Ranking Accuracy**: LightGBM achieves an **NDCG@10 approaching ~0.99** on offline validation sets.

---

## Project Structure

```text
recommendation-system/
├── api/                     # FastAPI backend
│   ├── cache.py             # Redis integration
│   ├── database.py          # SQLAlchemy setup
│   ├── model_registry.py    # HF Hub downloader & memory loader
│   ├── routes.py            # API endpoints (recommend, similar, feedback)
│   └── main.py              # Application entrypoint
├── frontend/                # Streamlit UI
│   └── app.py               # Main dashboard
├── models/                  # ML architectures & data structures
│   ├── two_tower.py         # PyTorch User & Item Embeddings
│   ├── ranker.py            # LightGBM LambdaRank integration
│   ├── feature_store.py     # Feature engineering & metadata store
│   └── faiss_index.py       # FAISS indexing logic
├── scripts/                 # Utility scripts
│   ├── load_data.py         # DB migration and data loader
│   ├── precompute_features.py 
│   └── upload_artifacts.py  # Hugging Face deployment script
├── training/                # Training loops & dataset definitions
│   ├── dataset.py
│   ├── train_ranker.py
│   └── train_two_tower.py
└── requirements.txt         # Python dependencies
```

---

## How to Use It

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory and add your credentials:
```env
REDIS_URL="redis://localhost:6379/0"
DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db"
DATABASE_URL_SYNC="postgresql://user:pass@localhost:5432/db"
HF_REPO_ID="your-username/recommender-system"
HF_TOKEN="your_hf_token"
```

### 3. Start the Backend (FastAPI)
The server will automatically download the required ML artifacts from Hugging Face on startup.
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- **Health Check**: `http://localhost:8000/health`
- **Swagger Docs**: `http://localhost:8000/docs`

### 4. Start the Frontend (Streamlit)
In a new terminal window:
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` to view the UI.
