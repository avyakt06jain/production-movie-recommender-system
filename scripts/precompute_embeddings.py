"""
Precompute item embeddings for FAISS index and Stage 2 features.

This script:
1. Loads the trained TwoTowerModel from artifacts/two_tower.pt
2. Loads features.pkl for item features
3. Runs the item tower over ALL movies in the catalog
4. L2-normalizes all embeddings
5. Saves artifacts/item_embeddings.npy (N, 64) float32
6. Saves artifacts/item_ids.npy — ordered list of movie_ids matching embedding rows

Usage:
    python scripts/precompute_embeddings.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.two_tower import TwoTowerModel
from training.dataset import load_features


CONFIG = {
    "model_path":       "artifacts/two_tower.pt",
    "features_path":    "artifacts/features.pkl",
    "embeddings_path":  "artifacts/item_embeddings.npy",
    "item_ids_path":    "artifacts/item_ids.npy",
    "batch_size":       512,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}


def load_two_tower_model(model_path: str, device: str = "cpu") -> TwoTowerModel:
    """Load a trained Two-Tower model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_users = checkpoint["n_users"]
    n_items = checkpoint["n_items"]
    embed_dim = checkpoint.get("embed_dim", 64)

    model = TwoTowerModel(n_users, n_items, embed_dim=embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    logger.info(
        f"Loaded Two-Tower model from {model_path} "
        f"(n_users={n_users}, n_items={n_items}, embed_dim={embed_dim})"
    )
    return model


def precompute_embeddings():
    """Main embedding precomputation pipeline."""
    logger.info("=" * 60)
    logger.info("Precomputing Item Embeddings")
    logger.info("=" * 60)

    device = torch.device(CONFIG["device"])
    logger.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_two_tower_model(CONFIG["model_path"], str(device))

    # ── Load features ──────────────────────────────────────────────────────
    features_data = load_features(CONFIG["features_path"])
    item_features = features_data["item_features"]

    # Get ordered list of all item IDs
    all_item_ids = sorted(item_features.keys())
    n_items = len(all_item_ids)
    logger.info(f"Processing {n_items} items")

    # ── Batch inference through item tower ─────────────────────────────────
    all_embeddings = []

    for start in range(0, n_items, CONFIG["batch_size"]):
        end = min(start + CONFIG["batch_size"], n_items)
        batch_ids = all_item_ids[start:end]

        item_batch = {
            "item_id": torch.tensor(
                [iid for iid in batch_ids],
                dtype=torch.long, device=device,
            ),
            "genre_vec": torch.tensor(
                np.array([item_features[iid]["genre_vec"] for iid in batch_ids]),
                dtype=torch.float32, device=device,
            ),
            "year_norm": torch.tensor(
                [item_features[iid]["year_norm"] for iid in batch_ids],
                dtype=torch.float32, device=device,
            ),
            "avg_rating_norm": torch.tensor(
                [item_features[iid].get("avg_rating_norm", item_features[iid]["avg_rating"] / 5.0) for iid in batch_ids],
                dtype=torch.float32, device=device,
            ),
            "log_count": torch.tensor(
                [item_features[iid].get("log_count", item_features[iid].get("rating_count_log", 0.0)) for iid in batch_ids],
                dtype=torch.float32, device=device,
            ),
        }

        with torch.no_grad():
            embeddings = model.item_tower(**item_batch)  # (batch, 64), already L2-normalized
            all_embeddings.append(embeddings.cpu().numpy())

        if (start // CONFIG["batch_size"]) % 5 == 0:
            logger.info(f"  Processed {end}/{n_items} items")

    # ── Concatenate and verify ─────────────────────────────────────────────
    item_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    item_ids_array = np.array(all_item_ids, dtype=np.int32)

    # Verify L2 normalization (should be ~1.0 for each row)
    norms = np.linalg.norm(item_embeddings, axis=1)
    logger.info(f"Embedding norms — min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")

    # Re-normalize just to be safe
    item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    logger.info(f"Item embeddings shape: {item_embeddings.shape}")
    logger.info(f"Item IDs shape: {item_ids_array.shape}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG["embeddings_path"]) or ".", exist_ok=True)

    np.save(CONFIG["embeddings_path"], item_embeddings)
    logger.info(f"Saved item embeddings to {CONFIG['embeddings_path']}")

    np.save(CONFIG["item_ids_path"], item_ids_array)
    logger.info(f"Saved item IDs to {CONFIG['item_ids_path']}")

    # ── Summary ────────────────────────────────────────────────────────────
    file_size_mb = os.path.getsize(CONFIG["embeddings_path"]) / (1024 * 1024)
    logger.info(f"Embeddings file size: {file_size_mb:.2f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    precompute_embeddings()
