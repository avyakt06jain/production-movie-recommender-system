"""
🎬 MovieRec — Streamlit Frontend

Three-page application:
  1. User Recommendations  — personalised movie recommendations for a user
  2. Movie Explorer         — find movies similar to a given movie
  3. System Stats           — health dashboard with model metadata
"""

import os

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🎬 MovieRec",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── global ────────────────────────────────────────── */
.block-container { padding-top: 1.5rem; }

/* ── movie cards ───────────────────────────────────── */
.movie-card {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.15s, box-shadow 0.15s;
    min-height: 200px;
}
.movie-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0f0f0;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.movie-rank {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}

/* ── genre badges ──────────────────────────────────── */
.genre-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 3px 2px 0;
    color: #fff;
}

/* ── score bar ─────────────────────────────────────── */
.score-bar-outer {
    background: rgba(255,255,255,0.1);
    border-radius: 6px;
    height: 8px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.score-bar-inner {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}

/* ── stat cards ────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.stat-value { font-size: 2rem; font-weight: 800; color: #667eea; }
.stat-label { font-size: 0.9rem; color: #aaa; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Genre → colour mapping
# ---------------------------------------------------------------------------
GENRE_COLORS: dict[str, str] = {
    "Action":       "#e74c3c", "Adventure":   "#e67e22", "Animation":     "#f1c40f",
    "Children's":   "#2ecc71", "Comedy":      "#1abc9c", "Crime":         "#9b59b6",
    "Documentary":  "#3498db", "Drama":       "#2980b9", "Fantasy":       "#8e44ad",
    "Film-Noir":    "#34495e", "Horror":      "#c0392b", "Musical":       "#e91e63",
    "Mystery":      "#607d8b", "Romance":     "#e84393", "Sci-Fi":        "#00bcd4",
    "Thriller":     "#ff5722", "War":         "#795548", "Western":       "#ff9800",
}


def _genre_badges_html(genres: list[str]) -> str:
    badges = []
    for g in genres:
        color = GENRE_COLORS.get(g, "#555")
        badges.append(f'<span class="genre-badge" style="background:{color}">{g}</span>')
    return "".join(badges)


def _score_bar_html(score: float) -> str:
    pct = max(0, min(100, score * 100))
    return (
        f'<div class="score-bar-outer">'
        f'<div class="score-bar-inner" style="width:{pct}%"></div>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#888;margin-top:2px">Score: {score:.3f}</div>'
    )


def _movie_card_html(rec: dict, show_rank: bool = True) -> str:
    rank_html = f'<span class="movie-rank">#{rec["rank"]}</span>' if show_rank else ""
    title = rec.get("title", f"Movie {rec['movie_id']}")
    genres = rec.get("genres", [])
    score = rec.get("score", 0.0)

    return (
        f'<div class="movie-card">'
        f'  {rank_html}'
        f'  <div class="movie-title">{title}</div>'
        f'  {_genre_badges_html(genres)}'
        f'  {_score_bar_html(score)}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None, timeout: float = 30.0) -> dict | None:
    """Make a GET request to the API backend."""
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200]
        st.error(f"API error ({exc.response.status_code}): {detail}")
        return None
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at **{API_BASE}**. "
            "Make sure the FastAPI server is running (`uvicorn api.main:app`)."
        )
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _api_post(path: str, json_body: dict, timeout: float = 10.0) -> dict | None:
    """Make a POST request to the API backend."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200]
        st.error(f"API error ({exc.response.status_code}): {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page: User Recommendations
# ---------------------------------------------------------------------------

def page_recommendations() -> None:
    st.title("🎬 Movie Recommendations")
    st.caption("Personalised recommendations powered by a multi-stage ML pipeline.")

    # ── Sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Parameters")
        user_id = st.number_input(
            "User ID", min_value=1, max_value=6040, value=42, step=1,
            help="Enter a user ID between 1 and 6040",
        )
        top_n = st.slider("Number of results", min_value=5, max_value=20, value=10)
        diversity = st.slider(
            "Diversity (λ)", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="0 = max diversity, 1 = max relevance",
        )
        exclude_watched = st.checkbox("Exclude watched movies", value=True)

        get_recs = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────────────────
    if get_recs:
        with st.spinner("Generating recommendations…"):
            data = _api_get(
                f"/api/v1/recommend/{user_id}",
                params={
                    "top_n": top_n,
                    "exclude_watched": str(exclude_watched).lower(),
                    "diversity": diversity,
                },
            )

        if data is None:
            return

        # Metadata bar
        col_cache, col_latency = st.columns(2)
        with col_cache:
            if data.get("cache_hit"):
                st.success("⚡ Cache hit")
            else:
                st.info("🔄 Fresh computation")
        with col_latency:
            st.metric("Latency", f"{data.get('latency_ms', 0):.0f} ms")

        st.divider()

        # Movie card grid (3 columns)
        recs = data.get("recommendations", [])
        if not recs:
            st.warning("No recommendations found for this user.")
            return

        cols = st.columns(3)
        for i, rec in enumerate(recs):
            with cols[i % 3]:
                st.markdown(_movie_card_html(rec, show_rank=True), unsafe_allow_html=True)
    else:
        st.info("👈 Set parameters in the sidebar and click **Get Recommendations**.")


# ---------------------------------------------------------------------------
# Page: Movie Explorer
# ---------------------------------------------------------------------------

def page_explorer() -> None:
    st.title("🔍 Movie Explorer")
    st.caption("Find movies similar to any movie in the catalog.")

    col_input, col_n = st.columns([2, 1])
    with col_input:
        movie_id = st.number_input(
            "Movie ID", min_value=1, max_value=4000, value=1, step=1,
            help="Enter a MovieLens movie ID",
        )
    with col_n:
        top_n = st.number_input("Results", min_value=1, max_value=50, value=10, step=1)

    search = st.button("🔎 Find Similar Movies", type="primary")

    if search:
        with st.spinner("Searching…"):
            data = _api_get(
                f"/api/v1/similar/{movie_id}",
                params={"top_n": top_n},
            )

        if data is None:
            return

        # Header
        st.subheader(f"Movies similar to: **{data.get('title', f'Movie {movie_id}')}**")

        col_cache, col_latency = st.columns(2)
        with col_cache:
            if data.get("cache_hit"):
                st.success("⚡ Cache hit")
            else:
                st.info("🔄 Fresh computation")
        with col_latency:
            st.metric("Latency", f"{data.get('latency_ms', 0):.0f} ms")

        st.divider()

        movies = data.get("similar_movies", [])
        if not movies:
            st.warning("No similar movies found.")
            return

        cols = st.columns(3)
        for i, movie in enumerate(movies):
            with cols[i % 3]:
                st.markdown(_movie_card_html(movie, show_rank=True), unsafe_allow_html=True)
    else:
        st.info("Enter a movie ID and click **Find Similar Movies**.")


# ---------------------------------------------------------------------------
# Page: System Stats
# ---------------------------------------------------------------------------

def page_stats() -> None:
    st.title("📊 System Stats")
    st.caption("Live health dashboard for the MovieRec API.")

    if st.button("🔄 Refresh", type="primary"):
        st.cache_data.clear()

    data = _api_get("/health")

    if data is None:
        return

    # Status cards
    c1, c2, c3 = st.columns(3)

    with c1:
        status = data.get("status", "unknown")
        color = "#2ecc71" if status == "ok" else "#e74c3c"
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value" style="color:{color}">{"✅" if status == "ok" else "❌"}</div>'
            f'<div class="stat-label">API Status</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        models_loaded = data.get("models_loaded", False)
        icon = "✅" if models_loaded else "⚠️"
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{icon}</div>'
            f'<div class="stat-label">Models Loaded</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c3:
        faiss_size = data.get("faiss_index_size", 0)
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{faiss_size:,}</div>'
            f'<div class="stat-label">FAISS Index Size</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Details table
    st.subheader("📋 Details")
    st.json(data)

    st.divider()

    # Architecture info
    st.subheader("🏗️ Architecture")
    st.markdown("""
    | Stage | Component | Description |
    |-------|-----------|-------------|
    | **1** | Two-Tower Neural Net + FAISS | Candidate generation — top-200 from full catalog |
    | **2** | LightGBM LambdaRank | Feature-rich ranking — scores each (user, item) pair |
    | **3** | MMR Re-ranking | Diversity + business rules — final top-N |
    """)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🎬 Recommendations": page_recommendations,
    "🔍 Movie Explorer": page_explorer,
    "📊 System Stats": page_stats,
}

with st.sidebar:
    st.markdown("---")
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page]()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_BASE}`")
st.sidebar.caption("MovieRec v1.0 • MovieLens 1M")
