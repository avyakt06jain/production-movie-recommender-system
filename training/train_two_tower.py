"""
Training script for the Two-Tower candidate generation model.

Uses BPR loss (Bayesian Personalized Ranking) with AdamW optimizer and
CosineAnnealingLR scheduler. Time-based train/val split with early stopping
on Recall@20.

Usage:
    python training/train_two_tower.py
"""

import os
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.two_tower import TwoTowerModel
from training.dataset import (
    MovieLensDataset,
    triplet_collate_fn,
    load_features,
    _encode_age_bucket,
    _encode_gender,
)


# ─── Training Configuration ──────────────────────────────────────────────────
CONFIG = {
    "batch_size":       2048,
    "lr":               1e-3,
    "weight_decay":     1e-5,
    "epochs":           30,
    "optimizer":        "AdamW",
    "scheduler":        "CosineAnnealingLR",
    "embed_dim":        64,
    "neg_sample_ratio": 4,
    "val_split":        0.1,     # time-based split (last 10% of timestamps = val)
    "patience":         5,       # early stopping on val Recall@20
    "model_save_path":  "artifacts/two_tower.pt",
    "features_path":    "artifacts/features.pkl",
    "eval_top_k":       [10, 20],
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    Encourages the model to rank positive items above negative items:
        L = -log(σ(pos_score - neg_score))

    Args:
        pos_scores: (B,) scores for positive (user, item) pairs
        neg_scores: (B,) scores for negative (user, item) pairs

    Returns:
        Scalar loss
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def compute_recall_at_k(
    model: TwoTowerModel,
    features_data: dict,
    eval_user_ids: list[int],
    k_values: list[int] = [10, 20],
    device: str = "cpu",
    max_eval_users: int = 1000,
) -> dict[str, float]:
    """
    Compute Recall@K for a set of users.

    For each user:
    1. Compute user embedding
    2. Score ALL items via dot product
    3. Exclude items the user has already rated
    4. Check if the user's positive items appear in the top-K

    Args:
        model:          Trained TwoTowerModel
        features_data:  Features dict from features.pkl
        eval_user_ids:  List of user IDs to evaluate
        k_values:       List of K values for Recall@K
        device:         Device to run on
        max_eval_users: Max users to evaluate (subsample for speed)

    Returns:
        Dict of {"recall@k": float} for each k
    """
    model.eval()

    user_features = features_data["user_features"]
    item_features = features_data["item_features"]
    user_positive_items = features_data["user_positive_items"]
    user_all_items = features_data["user_all_items"]
    all_item_ids = features_data["all_item_ids"]

    # Subsample users for speed if needed
    if len(eval_user_ids) > max_eval_users:
        eval_user_ids = list(np.random.choice(eval_user_ids, max_eval_users, replace=False))

    # Precompute all item embeddings
    item_ids_list = [iid for iid in all_item_ids if iid in item_features]
    item_id_tensor = torch.tensor([iid for iid in item_ids_list], dtype=torch.long, device=device)
    item_genre_vecs = torch.tensor(
        np.array([item_features[iid]["genre_vec"] for iid in item_ids_list]),
        dtype=torch.float32, device=device,
    )
    item_year_norms = torch.tensor(
        [item_features[iid]["year_norm"] for iid in item_ids_list],
        dtype=torch.float32, device=device,
    )
    item_avg_ratings = torch.tensor(
        [item_features[iid].get("avg_rating_norm", item_features[iid]["avg_rating"] / 5.0) for iid in item_ids_list],
        dtype=torch.float32, device=device,
    )
    item_log_counts = torch.tensor(
        [item_features[iid]["log_count"] for iid in item_ids_list],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        item_batch = {
            "item_id": item_id_tensor,
            "genre_vec": item_genre_vecs,
            "year_norm": item_year_norms,
            "avg_rating_norm": item_avg_ratings,
            "log_count": item_log_counts,
        }
        all_item_embs = model.item_tower(**item_batch)  # (N_items, 64)

    # Create a map from item_id → index in item_ids_list
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids_list)}

    max_k = max(k_values)
    recall_accum = defaultdict(float)
    valid_users = 0

    for uid in eval_user_ids:
        if uid not in user_features:
            continue
        pos_items = set(user_positive_items.get(uid, []))
        if not pos_items:
            continue

        uf = user_features[uid]
        with torch.no_grad():
            user_batch = {
                "user_id": torch.tensor([uf["user_id"]], dtype=torch.long, device=device),
                "age_bucket": torch.tensor([_encode_age_bucket(uf["age"])], dtype=torch.long, device=device),
                "gender": torch.tensor([_encode_gender(uf["gender"])], dtype=torch.long, device=device),
                "occupation": torch.tensor([uf["occupation"]], dtype=torch.long, device=device),
                "watched_genre_vec": torch.tensor([uf["watched_genre_vec"]], dtype=torch.float32, device=device),
            }
            user_emb = model.user_tower(**user_batch)  # (1, 64)

        # Score all items
        scores = (user_emb @ all_item_embs.T).squeeze(0)  # (N_items,)

        # Mask out already-rated items (keep only un-rated for ranking)
        rated_items = user_all_items.get(uid, set())
        for iid in rated_items:
            if iid in item_id_to_idx:
                scores[item_id_to_idx[iid]] = -float("inf")

        # Get top-K
        top_indices = torch.topk(scores, min(max_k, len(scores)), dim=0).indices.cpu().numpy()
        top_item_ids = [item_ids_list[i] for i in top_indices]

        for k in k_values:
            top_k_set = set(top_item_ids[:k])
            hits = len(pos_items & top_k_set)
            recall_accum[k] += hits / len(pos_items)

        valid_users += 1

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_accum[k] / max(valid_users, 1)

    model.train()
    return results


def time_based_split(features_data: dict, val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    """
    Split users into train/val based on the timestamp of their interactions.

    We use the ratings_df if available; otherwise we split by user_id ordering.
    The last `val_ratio` fraction of ratings (by timestamp) determines val users.

    Returns:
        (train_user_ids, val_user_ids)
    """
    if "ratings_df" in features_data and features_data["ratings_df"] is not None:
        import pandas as pd
        ratings_df = features_data["ratings_df"]

        # Sort by timestamp
        ratings_sorted = ratings_df.sort_values("timestamp")
        split_idx = int(len(ratings_sorted) * (1 - val_ratio))

        train_ratings = ratings_sorted.iloc[:split_idx]
        val_ratings = ratings_sorted.iloc[split_idx:]

        train_users = set(train_ratings["user_id"].unique())
        val_users = set(val_ratings["user_id"].unique())

        # Val users must also appear in train (to have learned embeddings)
        val_users = val_users & train_users
        # Exclude val users from train set for clean evaluation
        # Actually, keep all users in train — val measures generalization on new interactions
        train_user_ids = sorted(train_users)
        val_user_ids = sorted(val_users)
    else:
        # Fallback: split by user_id
        all_users = sorted(features_data["user_positive_items"].keys())
        split_idx = int(len(all_users) * (1 - val_ratio))
        train_user_ids = all_users[:split_idx]
        val_user_ids = all_users[split_idx:]

    logger.info(f"Time-based split: {len(train_user_ids)} train users, {len(val_user_ids)} val users")
    return train_user_ids, val_user_ids


def train():
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("Training Two-Tower Model")
    logger.info("=" * 60)
    logger.info(f"Config: {CONFIG}")

    device = torch.device(CONFIG["device"])
    logger.info(f"Device: {device}")

    # ── Load features ──────────────────────────────────────────────────────
    features_data = load_features(CONFIG["features_path"])
    n_users = max(features_data["user_features"].keys()) + 1
    n_items = max(features_data["item_features"].keys()) + 1
    logger.info(f"Vocabulary sizes: n_users={n_users}, n_items={n_items}")

    # ── Train / Val split ──────────────────────────────────────────────────
    train_user_ids, val_user_ids = time_based_split(features_data, CONFIG["val_split"])

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = MovieLensDataset(
        features_data,
        user_ids=train_user_ids,
        neg_sample_ratio=CONFIG["neg_sample_ratio"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=triplet_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = TwoTowerModel(n_users, n_items, embed_dim=CONFIG["embed_dim"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_recall_20 = 0.0
    patience_counter = 0
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for user_batch, pos_batch, neg_batch in train_loader:
            # Move to device
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            pos_batch = {k: v.to(device) for k, v in pos_batch.items()}
            neg_batch = {k: v.to(device) for k, v in neg_batch.items()}

            # Forward
            pos_scores = model(user_batch, pos_batch)
            neg_scores = model(user_batch, neg_batch)
            loss = bpr_loss(pos_scores, neg_scores)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ── Validation ─────────────────────────────────────────────────────
        recall_metrics = compute_recall_at_k(
            model, features_data, val_user_ids,
            k_values=CONFIG["eval_top_k"],
            device=str(device),
            max_eval_users=1000,
        )

        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Recall@10: {recall_metrics.get('recall@10', 0):.4f} | "
            f"Recall@20: {recall_metrics.get('recall@20', 0):.4f} | "
            f"LR: {lr_current:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Early stopping on Recall@20 ────────────────────────────────────
        recall_20 = recall_metrics.get("recall@20", 0)
        if recall_20 > best_recall_20:
            best_recall_20 = recall_20
            patience_counter = 0

            # Save best model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "n_users": n_users,
                "n_items": n_items,
                "embed_dim": CONFIG["embed_dim"],
                "config": CONFIG,
                "epoch": epoch,
                "best_recall_20": best_recall_20,
                "recall_metrics": recall_metrics,
            }
            torch.save(checkpoint, CONFIG["model_save_path"])
            logger.info(f"  ✓ New best model saved (Recall@20: {best_recall_20:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{CONFIG['patience']})")

            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch}. Best Recall@20: {best_recall_20:.4f}")
                break

    logger.info("=" * 60)
    logger.info(f"Training complete. Best Recall@20: {best_recall_20:.4f}")
    logger.info(f"Model saved to {CONFIG['model_save_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
