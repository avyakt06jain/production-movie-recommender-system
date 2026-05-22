"""
Model Registry — loads and holds all model artifacts in memory.

On startup, the registry checks the local `artifacts/` directory for model files.
If any are missing, it attempts to download them from Hugging Face Hub (gracefully
handles missing credentials for local dev where artifacts already exist).
"""

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Add project root to path so we can import from models/, retrieval/, features/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from features.feature_store import FeatureStore  # noqa: E402
from models.two_tower import TwoTowerModel  # noqa: E402
from retrieval.faiss_index import FAISSItemIndex  # noqa: E402

# Canonical list of the 18 MovieLens genres
GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Required artifact filenames
ARTIFACT_FILES: list[str] = [
    "two_tower.pt",
    "lgbm_ranker.txt",
    "item_embeddings.npy",
    "item_ids.npy",
    "features.pkl",
]


class ModelRegistry:
    """Loads and holds all model artifacts in memory."""

    def __init__(self) -> None:
        self.two_tower: TwoTowerModel | None = None
        self.ranker: lgb.Booster | None = None
        self.faiss_index: FAISSItemIndex | None = None
        self.feature_store: FeatureStore | None = None
        self.item_meta: pd.DataFrame | None = None  # movie_id, title, genres
        self.n_users: int = 0
        self.n_items: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def load_all(self, artifacts_dir: str = "artifacts") -> None:
        """Load all artifacts from disk. Downloads from HF Hub if missing."""
        artifacts_dir = os.path.abspath(artifacts_dir)
        os.makedirs(artifacts_dir, exist_ok=True)

        # Step 1: Attempt HF download for any missing files
        self._download_missing(artifacts_dir)

        # Step 2: Load features.pkl → FeatureStore
        self._load_feature_store(artifacts_dir)

        # Step 3: Derive n_users / n_items from the feature store
        self.n_users = len(self.feature_store.user_features)
        self.n_items = len(self.feature_store.item_features)
        logger.info(f"Feature store loaded — {self.n_users} users, {self.n_items} items")

        # Step 4: Load two_tower.pt → TwoTowerModel (eval mode, CPU)
        self._load_two_tower(artifacts_dir)

        # Step 5: Load item_embeddings.npy + item_ids.npy → FAISSItemIndex
        self._load_faiss_index(artifacts_dir)

        # Step 6: Load lgbm_ranker.txt → lgb.Booster
        self._load_ranker(artifacts_dir)

        # Step 7: Build item_meta DataFrame from feature_store
        self._build_item_meta()

        logger.info("✅ All model artifacts loaded successfully")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _download_missing(self, artifacts_dir: str) -> None:
        """Try downloading missing artifacts from Hugging Face Hub."""
        missing = [
            f for f in ARTIFACT_FILES
            if not os.path.exists(os.path.join(artifacts_dir, f))
        ]
        if not missing:
            logger.info("All artifact files present locally — skipping HF download")
            return

        logger.info(f"Missing artifacts: {missing}. Attempting HF Hub download…")
        repo_id = os.environ.get("HF_REPO_ID")
        token = os.environ.get("HF_TOKEN")

        if not repo_id:
            logger.warning(
                "HF_REPO_ID not set — cannot download artifacts. "
                "Ensure all files exist locally in the artifacts/ directory."
            )
            return

        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            for fname in missing:
                dest = os.path.join(artifacts_dir, fname)
                logger.info(f"Downloading {fname} from {repo_id}…")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=artifacts_dir,
                    token=token,
                )
                if os.path.exists(dest):
                    logger.info(f"  ✓ {fname} downloaded")
                else:
                    logger.warning(f"  ✗ {fname} download may have failed")
        except ImportError:
            logger.warning("huggingface_hub not installed — skipping artifact download")
        except Exception as exc:
            logger.warning(
                f"Could not download artifacts from HF Hub: {exc}. "
                "Continuing with whatever is available locally."
            )

    def _load_feature_store(self, artifacts_dir: str) -> None:
        """Load features.pkl into a FeatureStore instance."""
        features_path = os.path.join(artifacts_dir, "features.pkl")
        if not os.path.exists(features_path):
            logger.warning(
                "features.pkl not found — creating empty FeatureStore. "
                "Run scripts/precompute_features.py to generate it."
            )
            self.feature_store = FeatureStore()
            return

        self.feature_store = FeatureStore()
        self.feature_store.load(features_path)
        logger.info(f"Loaded features.pkl ({os.path.getsize(features_path) / 1e6:.1f} MB)")

    def _load_two_tower(self, artifacts_dir: str) -> None:
        """Load the TwoTowerModel checkpoint."""
        model_path = os.path.join(artifacts_dir, "two_tower.pt")
        if not os.path.exists(model_path):
            logger.warning(
                "two_tower.pt not found — TwoTower model will be unavailable. "
                "Run training/train_two_tower.py to create it."
            )
            return

        # The checkpoint stores model hyper-params alongside the state dict
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            n_users = checkpoint.get("n_users", self.n_users)
            n_items = checkpoint.get("n_items", self.n_items)
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            n_users = checkpoint.get("n_users", self.n_users)
            n_items = checkpoint.get("n_items", self.n_items)
            state_dict = checkpoint["state_dict"]
        else:
            # Assume checkpoint *is* the state dict
            n_users = self.n_users
            n_items = self.n_items
            state_dict = checkpoint

        self.two_tower = TwoTowerModel(n_users=n_users, n_items=n_items)
        self.two_tower.load_state_dict(state_dict)
        self.two_tower.eval()
        logger.info(f"Loaded TwoTowerModel ({n_users} users, {n_items} items)")

    def _load_faiss_index(self, artifacts_dir: str) -> None:
        """Load pre-computed item embeddings into a FAISS index."""
        emb_path = os.path.join(artifacts_dir, "item_embeddings.npy")
        ids_path = os.path.join(artifacts_dir, "item_ids.npy")

        if not os.path.exists(emb_path) or not os.path.exists(ids_path):
            logger.warning(
                "item_embeddings.npy / item_ids.npy not found — FAISS index "
                "will be unavailable. Run scripts/precompute_embeddings.py."
            )
            return

        item_embeddings = np.load(emb_path).astype(np.float32)
        item_ids = np.load(ids_path).tolist()

        dim = item_embeddings.shape[1]
        self.faiss_index = FAISSItemIndex(dim=dim)
        self.faiss_index.build(item_embeddings, item_ids)
        logger.info(
            f"FAISS index built — {len(item_ids)} items, {dim}-dim embeddings"
        )

    def _load_ranker(self, artifacts_dir: str) -> None:
        """Load the LightGBM ranker model."""
        ranker_path = os.path.join(artifacts_dir, "lgbm_ranker.txt")
        if not os.path.exists(ranker_path):
            logger.warning(
                "lgbm_ranker.txt not found — ranker will be unavailable. "
                "Run training/train_ranker.py to create it."
            )
            return

        self.ranker = lgb.Booster(model_file=ranker_path)
        logger.info("Loaded LightGBM ranker")

    def _build_item_meta(self) -> None:
        """Build item_meta DataFrame from the feature store for quick lookups."""
        if self.feature_store is None or not self.feature_store.item_features:
            self.item_meta = pd.DataFrame(columns=["movie_id", "title", "genres"])
            return

        rows = []
        for movie_id, feats in self.feature_store.item_features.items():
            genre_vec = feats.get("genre_vec", np.zeros(len(GENRES)))
            genre_list = [GENRES[i] for i, v in enumerate(genre_vec) if v > 0]
            rows.append(
                {
                    "movie_id": int(movie_id),
                    "title": feats.get("title", f"Movie {movie_id}"),
                    "genres": genre_list,
                    "avg_rating": feats.get("avg_rating", 0.0),
                    "rating_count": feats.get("rating_count", 0),
                    "year": feats.get("year", 0),
                }
            )

        self.item_meta = pd.DataFrame(rows).set_index("movie_id")
        logger.info(f"Built item_meta DataFrame with {len(self.item_meta)} movies")
