"""
LightGBM LambdaRank ranker for Stage 2 ranking.

Uses the lambdarank objective which directly optimizes NDCG — the standard
metric for ranking quality. Features include user/item static features,
two-tower scores from Stage 1, and interaction features like genre overlap.
"""

import os
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
from loguru import logger


# ─── LightGBM hyperparameters ─────────────────────────────────────────────────
LGBM_PARAMS: dict = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [5, 10, 20],
    "num_leaves":        63,
    "learning_rate":     0.05,
    "n_estimators":      500,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 20,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "verbose":           -1,
}


def train_ranker(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    params: Optional[dict] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """
    Train a LightGBM LambdaRank model.

    Args:
        train_features: (N_train, F) feature matrix
        train_labels:   (N_train,) relevance labels (0 or 1)
        train_groups:   (G_train,) number of items per group/query (user)
        val_features:   (N_val, F) feature matrix
        val_labels:     (N_val,) relevance labels
        val_groups:     (G_val,) number of items per group/query
        params:         LightGBM parameters (defaults to LGBM_PARAMS)
        num_boost_round: max number of boosting rounds
        early_stopping_rounds: stop if no improvement for this many rounds

    Returns:
        Trained lgb.Booster model
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    # Remove n_estimators from params since we use num_boost_round
    params_for_train = {k: v for k, v in params.items() if k != "n_estimators"}

    train_dataset = lgb.Dataset(
        data=train_features,
        label=train_labels,
        group=train_groups,
        free_raw_data=False,
    )
    val_dataset = lgb.Dataset(
        data=val_features,
        label=val_labels,
        group=val_groups,
        reference=train_dataset,
        free_raw_data=False,
    )

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
    ]

    logger.info(
        f"Training LightGBM ranker: {train_features.shape[0]} train samples, "
        f"{val_features.shape[0]} val samples, {train_features.shape[1]} features"
    )
    logger.info(f"Train groups: {len(train_groups)}, Val groups: {len(val_groups)}")

    model = lgb.train(
        params=params_for_train,
        train_set=train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best val NDCG@5: {model.best_score.get('val', {}).get('ndcg@5', 'N/A')}")
    logger.info(f"Best val NDCG@10: {model.best_score.get('val', {}).get('ndcg@10', 'N/A')}")
    logger.info(f"Best val NDCG@20: {model.best_score.get('val', {}).get('ndcg@20', 'N/A')}")

    return model


def predict_scores(model: lgb.Booster, features: np.ndarray) -> np.ndarray:
    """
    Predict ranking scores for a batch of (user, item) feature vectors.

    Args:
        model:    Trained LightGBM Booster
        features: (N, F) feature matrix

    Returns:
        (N,) array of predicted relevance scores (higher = more relevant)
    """
    return model.predict(features, num_iteration=model.best_iteration)


def save_ranker(model: lgb.Booster, path: str) -> None:
    """Save a trained LightGBM model to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_model(path, num_iteration=model.best_iteration)
    logger.info(f"Saved LightGBM ranker to {path}")


def load_ranker(path: str) -> lgb.Booster:
    """Load a trained LightGBM model from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Ranker model not found at {path}")
    model = lgb.Booster(model_file=path)
    logger.info(f"Loaded LightGBM ranker from {path} ({model.num_trees()} trees)")
    return model
