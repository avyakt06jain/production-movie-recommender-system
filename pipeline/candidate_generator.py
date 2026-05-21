"""
Stage 1: Candidate Generator — wraps Two-Tower model + FAISS index.

Given a user_id, produces the user embedding via the user tower and
queries the FAISS index for the top-K most similar items.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.two_tower import TwoTowerModel
from retrieval.faiss_index import FAISSItemIndex
from training.dataset import _encode_age_bucket, _encode_gender


class CandidateGenerator:
    """
    Stage 1 of the recommendation pipeline.

    Uses the Two-Tower model's user tower to compute a user embedding,
    then queries the FAISS inner-product index for the nearest items.
    Returns up to top_k (movie_id, score) pairs.
    """

    def __init__(
        self,
        two_tower_model: TwoTowerModel,
        faiss_index: FAISSItemIndex,
        feature_store: dict,
        device: str = "cpu",
    ):
        """
        Args:
            two_tower_model: Trained TwoTowerModel (will be put in eval mode)
            faiss_index:     Built FAISSItemIndex with all item embeddings
            feature_store:   Features dict (or object) with user_features sub-dict
            device:          Device for model inference
        """
        self.model = two_tower_model
        self.model.eval()
        self.faiss_index = faiss_index
        self.feature_store = feature_store
        self.device = device

    def _get_user_features(self, user_id: int) -> dict:
        """Retrieve user features from the feature store."""
        # Support both dict-style and object-style feature stores
        if isinstance(self.feature_store, dict):
            user_features = self.feature_store.get("user_features", self.feature_store)
            if user_id not in user_features:
                raise ValueError(f"User {user_id} not found in feature store")
            return user_features[user_id]
        else:
            # If feature_store is an object with get_user_features method
            return self.feature_store.get_user_features(user_id)

    def _compute_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Compute the 64-dim user embedding via the user tower.

        Returns:
            (64,) numpy array, L2-normalized
        """
        uf = self._get_user_features(user_id)

        user_batch = {
            "user_id": torch.tensor([uf["user_id"]], dtype=torch.long, device=self.device),
            "age_bucket": torch.tensor(
                [_encode_age_bucket(uf["age"])], dtype=torch.long, device=self.device
            ),
            "gender": torch.tensor(
                [_encode_gender(uf["gender"])], dtype=torch.long, device=self.device
            ),
            "occupation": torch.tensor(
                [uf["occupation"]], dtype=torch.long, device=self.device
            ),
            "watched_genre_vec": torch.tensor(
                [uf["watched_genre_vec"]], dtype=torch.float32, device=self.device
            ),
        }

        with torch.no_grad():
            user_emb = self.model.user_tower(**user_batch)  # (1, 64)

        return user_emb.cpu().numpy().squeeze(0)  # (64,)

    def generate(
        self,
        user_id: int,
        top_k: int = 200,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Generate candidate items for a user.

        Args:
            user_id:     The target user
            top_k:       Number of candidates to retrieve
            exclude_ids: Optional set of movie_ids to exclude (e.g., already watched)

        Returns:
            List of (movie_id, score) tuples sorted by descending similarity.
            If exclude_ids is provided, may return fewer than top_k items.
        """
        user_emb = self._compute_user_embedding(user_id)

        # Request more candidates if we're filtering
        fetch_k = top_k
        if exclude_ids:
            fetch_k = min(top_k + len(exclude_ids), self.faiss_index.size)

        candidates = self.faiss_index.search(user_emb, top_k=fetch_k)

        if exclude_ids:
            candidates = [(mid, score) for mid, score in candidates if mid not in exclude_ids]

        return candidates[:top_k]

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Public method to get a user's embedding (for use by other pipeline stages).

        Returns:
            (64,) numpy array, L2-normalized
        """
        return self._compute_user_embedding(user_id)
