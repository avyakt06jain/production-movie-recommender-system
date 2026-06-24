"""
Movie Recommender — Streamlit Frontend
Simplified UI version using native Streamlit components.
"""

import os
import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.write("A machine learning pipeline built with PyTorch, LightGBM, and FastAPI.")
st.divider()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API Request failed: {exc}. Is the FastAPI server running?")
        return None

# ---------------------------------------------------------------------------
# Layout (Tabs)
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🍿 For You (Recommendations)", "🔍 Movie Explorer", "⚙️ System Stats"])

# ── Tab 1: Recommendations ──────────────────────────────────────────────
with tab1:
    st.subheader("Generate Personalized Recommendations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        user_id = st.number_input("User ID (1-6040)", min_value=1, max_value=6040, value=42, step=1)
    with col2:
        top_n = st.number_input("Number of results", min_value=5, max_value=50, value=10, step=1)
    with col3:
        diversity = st.slider("MMR Diversity (0=Diverse, 1=Relevant)", min_value=0.0, max_value=1.0, value=0.7)
        
    exclude_watched = st.checkbox("Exclude movies the user has already watched", value=True)

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Fetching..."):
            data = api_get(
                f"/api/v1/recommend/{user_id}",
                params={
                    "top_n": top_n,
                    "exclude_watched": str(exclude_watched).lower(),
                    "diversity": diversity,
                },
            )
            
        if data:
            c1, c2 = st.columns(2)
            c1.info(f"Latency: {data.get('latency_ms', 0):.1f} ms")
            c2.info(f"Cache Hit: {data.get('cache_hit')}")
            
            recs = data.get("recommendations", [])
            if recs:
                # Use native Streamlit containers for a clean, simple look
                cols = st.columns(3)
                for i, rec in enumerate(recs):
                    with cols[i % 3]:
                        with st.container(border=True):
                            st.write(f"**#{rec['rank']} - {rec['title']}**")
                            st.caption(" • ".join(rec.get("genres", [])))
                            st.progress(max(0.0, min(1.0, rec['score'])), text=f"Score: {rec['score']:.3f}")
            else:
                st.warning("No recommendations found.")

# ── Tab 2: Similar Movies ───────────────────────────────────────────────
with tab2:
    st.subheader("Find Similar Movies using FAISS embeddings")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_movie_id = st.number_input("Movie ID", min_value=1, max_value=4000, value=1, step=1)
    with col2:
        sim_top_n = st.number_input("Top N Results", min_value=1, max_value=50, value=6, step=1)

    if st.button("Search Similar Movies", type="primary"):
        with st.spinner("Searching..."):
            data = api_get(f"/api/v1/similar/{sim_movie_id}", params={"top_n": sim_top_n})
            
        if data:
            st.success(f"Showing movies similar to: **{data.get('title')}**")
            
            sim_movies = data.get("similar_movies", [])
            if sim_movies:
                cols = st.columns(3)
                for i, rec in enumerate(sim_movies):
                    with cols[i % 3]:
                        with st.container(border=True):
                            st.write(f"**{rec['title']}**")
                            st.caption(" • ".join(rec.get("genres", [])))
            else:
                st.warning("No similar movies found.")

# ── Tab 3: System Health ────────────────────────────────────────────────
with tab3:
    st.subheader("API Status & Metadata")
    
    if st.button("Refresh Status"):
        st.cache_data.clear()
        
    data = api_get("/health")
    if data:
        st.json(data)
        
        st.markdown("### Architecture Pipeline")
        st.markdown("- **Stage 1**: Two-Tower Neural Network (PyTorch) + FAISS Vector Search")
        st.markdown("- **Stage 2**: LightGBM LambdaRank for precise feature scoring")
        st.markdown("- **Stage 3**: MMR (Maximal Marginal Relevance) for diversity re-ranking")
