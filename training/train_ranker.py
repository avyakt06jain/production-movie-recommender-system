"""
Training script for the LightGBM LambdaRank ranker (Stage 2).

This script:
1. Loads features.pkl and interaction data
2. Loads the trained Two-Tower model to compute two_tower_score features
3. Builds ~45 feature vectors per (user, item) pair
4. Positive samples: ratings ≥ 4.0 (label=1), negative: 10 random unrated items (label=0)
5. Time-based train/val split
6. Trains LightGBM with lambdarank objective
7. Saves the model to artifacts/lgbm_ranker.txt

Usage:
    python training/train_ranker.py
"""

import os
import re
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ranker import train_ranker, save_ranker, LGBM_PARAMS
from models.two_tower import TwoTowerModel
from training.dataset import (
    load_features,
    _encode_age_bucket,
    _encode_gender,
    GENRE_LIST,
    N_GENRES,
)


# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "features_path":     "artifacts/features.pkl",
    "two_tower_path":    "artifacts/two_tower.pt",
    "model_save_path":   "artifacts/lgbm_ranker.txt",
    "neg_samples_per_user": 10,
    "val_split":         0.1,     # time-based: last 10% of timestamps
    "num_boost_round":   500,
    "early_stopping":    50,
}


def _is_sequel(title: str) -> int:
    """Heuristic: check if a movie title suggests it is a sequel."""
    if not title:
        return 0
    # Roman numerals II, III, IV, V, VI, VII, VIII, IX, X
    roman_pattern = r'\b(II|III|IV|VI{0,3}|IX|X)\b'
    # Digits like "2", "3", etc. (but not years in parentheses)
    digit_pattern = r'(?<!\()\b[2-9]\b(?!\d{3}\))'
    if re.search(roman_pattern, title) or re.search(digit_pattern, title):
        return 1
    return 0


def load_two_tower_model(model_path: str, device: str = "cpu") -> TwoTowerModel:
    """Load a trained Two-Tower model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_users = checkpoint["n_users"]
    n_items = checkpoint["n_items"]
    embed_dim = checkpoint.get("embed_dim", 64)

    model = TwoTowerModel(n_users, n_items, embed_dim=embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded Two-Tower model from {model_path} (epoch {checkpoint.get('epoch', '?')}) to {device}")
    return model


def compute_two_tower_scores(
    model: TwoTowerModel,
    user_features: dict,
    item_features: dict,
    pairs: list[tuple[int, int]],
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict[tuple[int, int], float]:
    """
    Compute two-tower cosine similarity scores for a list of (user_id, item_id) pairs.

    Returns:
        Dict mapping (user_id, item_id) → score
    """
    model.eval()
    scores_dict = {}

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        uids = [p[0] for p in batch_pairs]
        iids = [p[1] for p in batch_pairs]

        # Build user tensors
        user_batch = {
            "user_id": torch.tensor([user_features[u]["user_id"] for u in uids], dtype=torch.long, device=device),
            "age_bucket": torch.tensor([_encode_age_bucket(user_features[u]["age"]) for u in uids], dtype=torch.long, device=device),
            "gender": torch.tensor([_encode_gender(user_features[u]["gender"]) for u in uids], dtype=torch.long, device=device),
            "occupation": torch.tensor([user_features[u]["occupation"] for u in uids], dtype=torch.long, device=device),
            "watched_genre_vec": torch.tensor(
                np.array([user_features[u]["watched_genre_vec"] for u in uids]),
                dtype=torch.float32, device=device,
            ),
        }

        # Build item tensors
        item_batch = {
            "item_id": torch.tensor([i for i in iids], dtype=torch.long, device=device),
            "genre_vec": torch.tensor(
                np.array([item_features[i]["genre_vec"] for i in iids]),
                dtype=torch.float32, device=device,
            ),
            "year_norm": torch.tensor([item_features[i]["year_norm"] for i in iids], dtype=torch.float32, device=device),
            "avg_rating_norm": torch.tensor(
                [item_features[i].get("avg_rating_norm", item_features[i]["avg_rating"] / 5.0) for i in iids],
                dtype=torch.float32, device=device,
            ),
            "log_count": torch.tensor([item_features[i].get("log_count", item_features[i].get("rating_count_log", 0.0)) for i in iids], dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            batch_scores = model(user_batch, item_batch).cpu().numpy()

        for idx, pair in enumerate(batch_pairs):
            scores_dict[pair] = float(batch_scores[idx])

    return scores_dict


def build_rank_features(
    user_id: int,
    item_id: int,
    user_features: dict,
    item_features: dict,
    two_tower_score: float,
    popularity_percentiles: dict[int, float],
    user_genre_rated_counts: dict[int, dict[int, int]] | None = None,
    user_first_genre_ts: dict[int, dict[int, float]] | None = None,
    max_timestamp: float = 1e9,
) -> np.ndarray:
    """
    Build the ~45-dim feature vector for a single (user, item) pair.

    Feature layout:
    [0]       user_age_norm
    [1]       user_gender_enc
    [2]       user_occupation
    [3]       user_avg_rating_given
    [4]       user_rating_count
    [5:23]    user_top_genres (18-dim)
    [23]      item_avg_rating
    [24]      item_rating_count_log
    [25]      item_year_norm
    [26:44]   item_genres (18-dim multi-hot)
    [44]      two_tower_score
    [45]      genre_overlap
    [46]      user_has_rated_genre (majority genre)
    [47]      popularity_percentile
    [48]      is_sequel
    """
    uf = user_features[user_id]
    ifeat = item_features[item_id]

    # User features
    user_age_norm = _encode_age_bucket(uf["age"]) / 7.0
    user_gender_enc = 1 if uf["gender"] == "M" else 0
    user_occupation = uf["occupation"]
    user_avg_rating = uf.get("avg_rating", 3.5)
    user_rating_count = uf.get("rating_count", 0)
    user_genre_vec = np.array(uf["watched_genre_vec"], dtype=np.float32)

    # Item features
    item_avg_rating = ifeat.get("avg_rating", 3.0)
    item_log_count = ifeat.get("log_count", ifeat.get("rating_count_log", 0.0))
    item_year_norm = ifeat["year_norm"]
    item_genre_vec = np.array(ifeat["genre_vec"], dtype=np.float32)

    # Interaction features
    genre_overlap = float(np.dot(user_genre_vec, item_genre_vec))

    # user_has_rated_genre: check if user has rated ≥3 movies in the item's majority genre
    user_has_rated_genre = 0
    if user_genre_rated_counts and user_id in user_genre_rated_counts:
        genre_counts = user_genre_rated_counts[user_id]
        for g_idx in range(N_GENRES):
            if item_genre_vec[g_idx] > 0 and genre_counts.get(g_idx, 0) >= 3:
                user_has_rated_genre = 1
                break

    popularity_pct = popularity_percentiles.get(item_id, 0.5)
    is_sequel = _is_sequel(ifeat.get("title", ""))

    features = np.concatenate([
        [user_age_norm, user_gender_enc, user_occupation, user_avg_rating, user_rating_count],
        user_genre_vec,                     # 18 dims
        [item_avg_rating, item_log_count, item_year_norm],
        item_genre_vec,                     # 18 dims
        [two_tower_score, genre_overlap, user_has_rated_genre, popularity_pct, is_sequel],
    ]).astype(np.float32)

    return features


def prepare_training_data(
    features_data: dict,
    two_tower_scores: dict[tuple[int, int], float],
    user_ids: list[int],
    neg_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (features, labels, groups) arrays for LightGBM training.

    Args:
        features_data:     Loaded features.pkl
        two_tower_scores:  Precomputed (user, item) → score dict
        user_ids:          Subset of user IDs to include
        neg_samples:       Number of negative samples per user

    Returns:
        features: (N, F) float32 array
        labels:   (N,) float32 array (0 or 1)
        groups:   (G,) array — count of rows per user (for lambdarank)
    """
    user_features = features_data["user_features"]
    item_features = features_data["item_features"]
    user_positive_items = features_data["user_positive_items"]
    user_all_items = features_data["user_all_items"]
    all_item_ids = features_data["all_item_ids"]

    # Compute popularity percentiles
    all_counts = []
    for iid in all_item_ids:
        if iid in item_features:
            all_counts.append((iid, item_features[iid].get("rating_count", 0)))
    all_counts.sort(key=lambda x: x[1])
    popularity_percentiles = {}
    for rank, (iid, _) in enumerate(all_counts):
        popularity_percentiles[iid] = rank / max(len(all_counts) - 1, 1)

    # Build user genre rated counts from user features
    user_genre_rated_counts: dict[int, dict[int, int]] = {}
    for uid in user_ids:
        if uid not in user_features:
            continue
        uf = user_features[uid]
        genre_vec = uf["watched_genre_vec"]
        count = uf.get("rating_count", 10)
        # Approximate per-genre counts from the genre preference vector
        genre_counts = {}
        for g_idx in range(N_GENRES):
            estimated = int(genre_vec[g_idx] * count)
            if estimated > 0:
                genre_counts[g_idx] = estimated
        user_genre_rated_counts[uid] = genre_counts

    all_features = []
    all_labels = []
    groups = []

    valid_items = set(item_features.keys())

    for uid in user_ids:
        if uid not in user_features:
            continue

        pos_items = [iid for iid in user_positive_items.get(uid, []) if iid in valid_items]
        if not pos_items:
            continue

        rated_items = user_all_items.get(uid, set())
        unrated = [iid for iid in all_item_ids if iid not in rated_items and iid in valid_items]

        # Sample negatives
        n_neg = min(neg_samples, len(unrated))
        neg_items = list(np.random.choice(unrated, size=n_neg, replace=False)) if unrated else []

        user_samples = []

        # Positives
        for iid in pos_items:
            score = two_tower_scores.get((uid, iid), 0.0)
            feat = build_rank_features(
                uid, iid, user_features, item_features, score,
                popularity_percentiles, user_genre_rated_counts,
            )
            user_samples.append((feat, 1.0))

        # Negatives
        for iid in neg_items:
            score = two_tower_scores.get((uid, iid), 0.0)
            feat = build_rank_features(
                uid, iid, user_features, item_features, score,
                popularity_percentiles, user_genre_rated_counts,
            )
            user_samples.append((feat, 0.0))

        if user_samples:
            for feat, label in user_samples:
                all_features.append(feat)
                all_labels.append(label)
            groups.append(len(user_samples))

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    groups = np.array(groups, dtype=np.int32)

    return features, labels, groups


def train():
    """Main LightGBM ranker training pipeline."""
    logger.info("=" * 60)
    logger.info("Training LightGBM Ranker")
    logger.info("=" * 60)

    t0 = time.time()

    # ── Load features ──────────────────────────────────────────────────────
    features_data = load_features(CONFIG["features_path"])

    # ── Load Two-Tower model for score computation ─────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    two_tower_model = None
    if Path(CONFIG["two_tower_path"]).exists():
        two_tower_model = load_two_tower_model(CONFIG["two_tower_path"], device=device)
        logger.info("Two-Tower model loaded for score computation")
    else:
        logger.warning(f"Two-Tower model not found at {CONFIG['two_tower_path']}, using 0.0 for two_tower_score")

    # ── Time-based split ───────────────────────────────────────────────────
    user_positive_items = features_data["user_positive_items"]
    all_users = sorted(user_positive_items.keys())

    if "ratings_df" in features_data and features_data["ratings_df"] is not None:
        ratings_df = features_data["ratings_df"]
        ratings_sorted = ratings_df.sort_values("timestamp")
        split_idx = int(len(ratings_sorted) * (1 - CONFIG["val_split"]))

        train_ratings = ratings_sorted.iloc[:split_idx]
        val_ratings = ratings_sorted.iloc[split_idx:]

        train_users = sorted(set(train_ratings["user_id"].unique()))
        val_users = sorted(set(val_ratings["user_id"].unique()) & set(train_ratings["user_id"].unique()))
    else:
        split_idx = int(len(all_users) * (1 - CONFIG["val_split"]))
        train_users = all_users[:split_idx]
        val_users = all_users[split_idx:]

    logger.info(f"Train users: {len(train_users)}, Val users: {len(val_users)}")

    # ── Collect all (user, item) pairs for two-tower scoring ───────────────
    logger.info("Collecting all (user, item) pairs for feature computation...")

    all_pairs: list[tuple[int, int]] = []
    valid_items = set(features_data["item_features"].keys())
    all_item_ids = features_data["all_item_ids"]

    for uid in train_users + val_users:
        if uid not in features_data["user_features"]:
            continue
        pos_items = [iid for iid in user_positive_items.get(uid, []) if iid in valid_items]
        rated_items = features_data["user_all_items"].get(uid, set())

        for iid in pos_items:
            all_pairs.append((uid, iid))

        # Also add some negatives for scoring (we'll sample more later)
        unrated = [iid for iid in all_item_ids if iid not in rated_items and iid in valid_items]
        n_neg = min(CONFIG["neg_samples_per_user"], len(unrated))
        if unrated:
            neg_sample = list(np.random.choice(unrated, size=n_neg, replace=False))
            for iid in neg_sample:
                all_pairs.append((uid, iid))

    logger.info(f"Computing two-tower scores for {len(all_pairs)} pairs...")

    if two_tower_model is not None:
        two_tower_scores = compute_two_tower_scores(
            two_tower_model,
            features_data["user_features"],
            features_data["item_features"],
            all_pairs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        two_tower_scores = {p: 0.0 for p in all_pairs}

    logger.info(f"Computed {len(two_tower_scores)} two-tower scores")

    # ── Build training data ────────────────────────────────────────────────
    logger.info("Building training feature matrix...")
    train_features, train_labels, train_groups = prepare_training_data(
        features_data, two_tower_scores, train_users,
        neg_samples=CONFIG["neg_samples_per_user"],
    )
    logger.info(f"Train: {train_features.shape[0]} samples, {train_features.shape[1]} features, "
                f"{len(train_groups)} groups, {train_labels.sum():.0f} positives")

    logger.info("Building validation feature matrix...")
    val_features, val_labels, val_groups = prepare_training_data(
        features_data, two_tower_scores, val_users,
        neg_samples=CONFIG["neg_samples_per_user"],
    )
    logger.info(f"Val: {val_features.shape[0]} samples, {val_features.shape[1]} features, "
                f"{len(val_groups)} groups, {val_labels.sum():.0f} positives")

    # ── Train ──────────────────────────────────────────────────────────────
    model = train_ranker(
        train_features, train_labels, train_groups,
        val_features, val_labels, val_groups,
        num_boost_round=CONFIG["num_boost_round"],
        early_stopping_rounds=CONFIG["early_stopping"],
    )

    # ── Log final metrics ──────────────────────────────────────────────────
    best_scores = model.best_score.get("val", {})
    logger.info(f"Final NDCG@5:  {best_scores.get('ndcg@5', 'N/A')}")
    logger.info(f"Final NDCG@10: {best_scores.get('ndcg@10', 'N/A')}")
    logger.info(f"Final NDCG@20: {best_scores.get('ndcg@20', 'N/A')}")

    # ── Feature importance ─────────────────────────────────────────────────
    feature_names = (
        ["user_age_norm", "user_gender_enc", "user_occupation", "user_avg_rating", "user_rating_count"]
        + [f"user_genre_{g}" for g in GENRE_LIST]
        + ["item_avg_rating", "item_rating_count_log", "item_year_norm"]
        + [f"item_genre_{g}" for g in GENRE_LIST]
        + ["two_tower_score", "genre_overlap", "user_has_rated_genre", "popularity_pct", "is_sequel"]
    )

    importance = model.feature_importance(importance_type="gain")
    if len(feature_names) == len(importance):
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 feature importances (gain):")
        for name, imp in feat_imp[:10]:
            logger.info(f"  {name}: {imp:.1f}")

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]) or ".", exist_ok=True)
    save_ranker(model, CONFIG["model_save_path"])

    elapsed = time.time() - t0
    logger.info(f"Total training time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
