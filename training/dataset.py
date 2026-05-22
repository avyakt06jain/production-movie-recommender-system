"""
PyTorch Dataset for Two-Tower model training with BPR triplet sampling.

Each sample is a triplet: (user_features, positive_item_features, negative_item_features)
where the positive item is one the user rated ≥4.0 and the negative item is one the user
has NOT rated. We use a 4:1 negative sampling ratio — each positive is paired with 4
random negatives across the epoch.
"""

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


# MovieLens 1M age buckets → index mapping
AGE_BUCKET_MAP: dict[int, int] = {
    1: 1,    # Under 18
    18: 2,   # 18-24
    25: 3,   # 25-34
    35: 4,   # 35-44
    45: 5,   # 45-49
    50: 6,   # 50-55
    56: 7,   # 56+
}

GENRE_LIST: list[str] = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GENRE_LIST)}
N_GENRES: int = len(GENRE_LIST)


def load_features(features_path: str = "artifacts/features.pkl") -> dict[str, Any]:
    """
    Load the precomputed features dictionary from disk.

    Expected structure:
    {
        "user_features": {
            user_id: {
                "user_id": int,
                "age": int,          # raw MovieLens age bucket value (1,18,25,35,45,50,56)
                "gender": str,       # "M" or "F"
                "occupation": int,   # 0-20
                "watched_genre_vec": np.ndarray (18,),  # avg genre vector of positive ratings
                "avg_rating": float,
                "rating_count": int,
            }
        },
        "item_features": {
            movie_id: {
                "movie_id": int,
                "genre_vec": np.ndarray (18,),  # multi-hot
                "year_norm": float,              # (year - 1920) / 100
                "avg_rating": float,
                "avg_rating_norm": float,        # normalized
                "rating_count": int,
                "log_count": float,              # log1p(rating_count)
                "genres": list[str],
                "title": str,
            }
        },
        "user_positive_items": {
            user_id: list[int]    # movie_ids with rating ≥ 4.0
        },
        "user_all_items": {
            user_id: set[int]     # all movie_ids the user has rated
        },
        "all_item_ids": list[int],   # all movie_ids in the catalog
        "ratings_df": ...,            # optional, pandas DataFrame of all ratings
    }
    """
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        f"Loaded features: {len(data.get('user_features', {}))} users, "
        f"{len(data.get('item_features', {}))} items"
    )
    return data


def _encode_age_bucket(age: int) -> int:
    """Convert raw MovieLens age bucket value to embedding index."""
    return AGE_BUCKET_MAP.get(age, 0)


def _encode_gender(gender: str) -> int:
    """Convert gender string to embedding index (0=pad, 1=F, 2=M)."""
    if gender == "F":
        return 1
    elif gender == "M":
        return 2
    return 0


def _make_user_tensors(user_feat: dict) -> dict[str, torch.Tensor]:
    """Convert a user feature dict to a dict of tensors (un-batched, no batch dim).

    Handles both raw format (age as MovieLens code, gender as 'M'/'F')
    and pre-encoded format (age_bucket as int, gender as int).
    """
    # Handle age: may be raw MovieLens code or pre-encoded bucket
    age_val = user_feat.get("age", user_feat.get("age_bucket", 0))
    if age_val in AGE_BUCKET_MAP:
        age_bucket = _encode_age_bucket(age_val)
    else:
        age_bucket = int(age_val)

    # Handle gender: may be 'M'/'F' string or pre-encoded int
    gender_val = user_feat.get("gender_raw", user_feat.get("gender", 0))
    if isinstance(gender_val, str):
        gender_enc = _encode_gender(gender_val)
    else:
        # Pre-encoded: 0=F, 1=M → shift to match embedding (0=pad, 1=F, 2=M)
        gender_enc = int(gender_val) + 1

    # Handle watched_genre_vec
    genre_vec = user_feat.get("watched_genre_vec", user_feat.get("genre_pref_vec", np.zeros(N_GENRES)))

    return {
        "user_id": torch.tensor(user_feat.get("user_id", 0), dtype=torch.long),
        "age_bucket": torch.tensor(age_bucket, dtype=torch.long),
        "gender": torch.tensor(gender_enc, dtype=torch.long),
        "occupation": torch.tensor(user_feat["occupation"], dtype=torch.long),
        "watched_genre_vec": torch.tensor(genre_vec, dtype=torch.float32),
    }


def _make_item_tensors(item_feat: dict, item_id: int = 0) -> dict[str, torch.Tensor]:
    """Convert an item feature dict to a dict of tensors (un-batched, no batch dim).

    Args:
        item_feat: Item feature dict from features.pkl
        item_id: The movie_id (passed separately since it's the dict key, not a value)
    """
    # movie_id may or may not be in the dict — use the explicit param as fallback
    mid = item_feat.get("movie_id", item_id)
    avg_rating = float(item_feat.get("avg_rating", 0.0) or 0.0)
    log_count = float(item_feat.get("log_count", item_feat.get("rating_count_log", 0.0)))
    return {
        "item_id": torch.tensor(mid, dtype=torch.long),
        "genre_vec": torch.tensor(item_feat["genre_vec"], dtype=torch.float32),
        "year_norm": torch.tensor(float(item_feat["year_norm"]), dtype=torch.float32),
        "avg_rating_norm": torch.tensor(item_feat.get("avg_rating_norm", avg_rating / 5.0), dtype=torch.float32),
        "log_count": torch.tensor(log_count, dtype=torch.float32),
    }


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for BPR triplet training of the Two-Tower model.

    Each __getitem__ returns a single (user, positive_item, negative_item) triplet.
    The effective dataset size is len(positive_interactions) * neg_sample_ratio,
    so each positive is paired with `neg_sample_ratio` different negatives across
    the epoch (negatives are sampled randomly on each call).
    """

    def __init__(
        self,
        features_data: dict[str, Any],
        user_ids: list[int] | None = None,
        neg_sample_ratio: int = 4,
    ):
        """
        Args:
            features_data: dict loaded from features.pkl via load_features()
            user_ids:       optional subset of user_ids to use (for train/val split)
            neg_sample_ratio: number of negatives per positive per epoch
        """
        super().__init__()

        self.user_features = features_data["user_features"]
        self.item_features = features_data["item_features"]
        self.user_positive_items = features_data["user_positive_items"]
        self.user_all_items = features_data["user_all_items"]
        self.all_item_ids = features_data["all_item_ids"]
        self.all_item_set = set(self.all_item_ids)
        self.neg_sample_ratio = neg_sample_ratio

        # Filter to requested users (for train/val splitting)
        if user_ids is not None:
            valid_users = set(user_ids)
        else:
            valid_users = set(self.user_positive_items.keys())

        # Build flat list of (user_id, positive_item_id) pairs
        self.samples: list[tuple[int, int]] = []
        for uid in valid_users:
            pos_items = self.user_positive_items.get(uid, [])
            for iid in pos_items:
                if iid in self.item_features:
                    # Repeat each positive neg_sample_ratio times
                    for _ in range(self.neg_sample_ratio):
                        self.samples.append((uid, iid))

        logger.info(
            f"Created MovieLensDataset with {len(self.samples)} triplet samples "
            f"({len(self.samples) // max(self.neg_sample_ratio, 1)} positives × "
            f"{self.neg_sample_ratio} negatives) for {len(valid_users)} users"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict, dict, dict]:
        """
        Returns:
            (user_tensors, pos_item_tensors, neg_item_tensors)
            Each is a dict of tensors without a batch dimension.
        """
        user_id, pos_item_id = self.samples[idx]

        # Sample a negative item the user has NOT interacted with
        user_rated = self.user_all_items.get(user_id, set())
        neg_item_id = random.choice(self.all_item_ids)
        # Rejection sampling (fast since catalog >> user history)
        while neg_item_id in user_rated:
            neg_item_id = random.choice(self.all_item_ids)

        user_tensors = _make_user_tensors(self.user_features[user_id])
        pos_tensors = _make_item_tensors(self.item_features[pos_item_id], item_id=pos_item_id)
        neg_tensors = _make_item_tensors(self.item_features[neg_item_id], item_id=neg_item_id)

        return user_tensors, pos_tensors, neg_tensors


def triplet_collate_fn(
    batch: list[tuple[dict, dict, dict]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Custom collate function to batch triplet samples.

    Takes a list of (user_dict, pos_item_dict, neg_item_dict) tuples and
    stacks each field into batched tensors.

    Returns:
        (user_batch, pos_item_batch, neg_item_batch) — each is a dict of
        batched tensors with shape (B, ...).
    """
    users, pos_items, neg_items = zip(*batch)

    def stack_dicts(dicts: tuple[dict, ...]) -> dict[str, torch.Tensor]:
        keys = dicts[0].keys()
        return {k: torch.stack([d[k] for d in dicts], dim=0) for k in keys}

    return stack_dicts(users), stack_dicts(pos_items), stack_dicts(neg_items)
