"""
FAISS index for approximate nearest neighbor search on item embeddings.

Since all embeddings are L2-normalized, inner-product search (IndexFlatIP)
is equivalent to cosine similarity. With only ~3,700 items, flat exact search
runs in sub-millisecond time — no need for approximate indexes like HNSW.
"""

from pathlib import Path

import faiss
import numpy as np
from loguru import logger


class FAISSItemIndex:
    """
    Stores all item embeddings in a FAISS flat inner-product index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # IP = inner product
        self.item_ids: list[int] = []
        self._id_to_idx: dict[int, int] = {}  # movie_id → row index for reverse lookup

    def build(self, item_embeddings: np.ndarray, item_ids: list[int]) -> None:
        """
        Build the index from precomputed item embeddings.

        Args:
            item_embeddings: (N, dim) float32 array, should be L2-normalized
            item_ids:        list of N movie_ids matching the embedding rows
        """
        assert item_embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: expected {self.dim}, got {item_embeddings.shape[1]}"
        )
        assert len(item_ids) == item_embeddings.shape[0], (
            f"Length mismatch: {len(item_ids)} ids vs {item_embeddings.shape[0]} embeddings"
        )

        # Reset index if rebuilding
        self.index = faiss.IndexFlatIP(self.dim)
        self.item_ids = list(item_ids)
        self._id_to_idx = {mid: idx for idx, mid in enumerate(self.item_ids)}

        self.index.add(item_embeddings.astype(np.float32))
        logger.info(f"Built FAISS index with {self.index.ntotal} items, dim={self.dim}")

    def search(self, user_embedding: np.ndarray, top_k: int = 200) -> list[tuple[int, float]]:
        """
        Find the top_k most similar items to a user embedding.

        Args:
            user_embedding: (dim,) or (1, dim) float32 array, L2-normalized
            top_k:          number of nearest neighbors to return

        Returns:
            List of (movie_id, score) tuples sorted by descending similarity
        """
        query = user_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, top_k)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx < 0:
                # FAISS returns -1 for missing results
                continue
            results.append((self.item_ids[idx], float(scores[0][j])))
        return results

    def get_item_embedding(self, movie_id: int) -> np.ndarray | None:
        """
        Return the stored embedding for a specific movie_id.

        Args:
            movie_id: the movie ID to look up

        Returns:
            (dim,) float32 array, or None if movie_id not in index
        """
        idx = self._id_to_idx.get(movie_id)
        if idx is None:
            return None
        # faiss.IndexFlatIP stores vectors; reconstruct retrieves them
        embedding = np.empty(self.dim, dtype=np.float32)
        self.index.reconstruct(idx, embedding)
        return embedding

    @property
    def size(self) -> int:
        """Number of items in the index."""
        return self.index.ntotal

    def save(self, index_path: str, ids_path: str) -> None:
        """Save the FAISS index and item IDs to disk."""
        faiss.write_index(self.index, str(index_path))
        np.save(ids_path, np.array(self.item_ids))
        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {index_path}")

    @classmethod
    def load(cls, index_path: str, ids_path: str, dim: int = 64) -> "FAISSItemIndex":
        """Load a FAISS index and item IDs from disk."""
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(ids_path).exists():
            raise FileNotFoundError(f"Item IDs not found at {ids_path}")

        obj = cls(dim=dim)
        obj.index = faiss.read_index(str(index_path))
        obj.item_ids = np.load(ids_path).tolist()
        obj._id_to_idx = {mid: idx for idx, mid in enumerate(obj.item_ids)}
        logger.info(f"Loaded FAISS index with {obj.index.ntotal} items from {index_path}")
        return obj
