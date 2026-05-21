"""
Two-Tower model for candidate generation (Stage 1).

Architecture:
  UserTower: user_id + age + gender + occupation + watched_genres → 64-dim L2-normalized embedding
  ItemTower: item_id + genres + year + avg_rating + log_count → 64-dim L2-normalized embedding

Training objective: BPR loss on dot-product of user/item embeddings.
At inference, embeddings are stored in a FAISS index for sub-ms ANN search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user features into a 64-dim L2-normalized embedding."""

    def __init__(self, n_users: int, n_occupations: int = 21, n_genres: int = 18, embed_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embed_dim)       # +1 for unknown/padding
        self.age_emb = nn.Embedding(8, 16)                          # 7 age buckets + 1 pad
        self.gender_emb = nn.Embedding(3, 4)                        # M/F + pad
        self.occ_emb = nn.Embedding(n_occupations + 1, 16)          # +1 for unknown
        self.genre_proj = nn.Linear(n_genres, 18)

        # 64 + 16 + 4 + 16 + 18 = 118
        input_dim = embed_dim + 16 + 4 + 16 + 18
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(
        self,
        user_id: torch.Tensor,
        age_bucket: torch.Tensor,
        gender: torch.Tensor,
        occupation: torch.Tensor,
        watched_genre_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_id:          (B,) LongTensor
            age_bucket:       (B,) LongTensor  (0-7)
            gender:           (B,) LongTensor  (0=pad, 1=F, 2=M)
            occupation:       (B,) LongTensor  (0-21)
            watched_genre_vec:(B, 18) FloatTensor — avg genre vector of positively rated movies

        Returns:
            (B, 64) L2-normalized user embedding
        """
        x = torch.cat([
            self.user_emb(user_id),
            self.age_emb(age_bucket),
            self.gender_emb(gender),
            self.occ_emb(occupation),
            self.genre_proj(watched_genre_vec),
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    """Encodes item features into a 64-dim L2-normalized embedding."""

    def __init__(self, n_items: int, n_genres: int = 18, embed_dim: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim)       # +1 for unknown/padding
        self.genre_proj = nn.Linear(n_genres, 32)
        self.year_proj = nn.Linear(1, 8)

        # 64 + 32 + 8 + 2 = 106  (+2 for avg_rating_norm, log_count)
        input_dim = embed_dim + 32 + 8 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(
        self,
        item_id: torch.Tensor,
        genre_vec: torch.Tensor,
        year_norm: torch.Tensor,
        avg_rating_norm: torch.Tensor,
        log_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_id:          (B,) LongTensor
            genre_vec:        (B, 18) FloatTensor — multi-hot genre encoding
            year_norm:        (B,) FloatTensor — (year - 1920) / 100
            avg_rating_norm:  (B,) FloatTensor — normalized global average rating
            log_count:        (B,) FloatTensor — log1p(rating_count)

        Returns:
            (B, 64) L2-normalized item embedding
        """
        x = torch.cat([
            self.item_emb(item_id),
            self.genre_proj(genre_vec),
            self.year_proj(year_norm.unsqueeze(-1)),
            avg_rating_norm.unsqueeze(-1),
            log_count.unsqueeze(-1),
        ], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower (dual encoder) model for candidate generation.

    Computes dot-product similarity between user and item embeddings.
    Since both towers produce L2-normalized vectors, the dot product
    equals cosine similarity.
    """

    def __init__(self, n_users: int, n_items: int, n_occupations: int = 21, n_genres: int = 18, embed_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(n_users, n_occupations=n_occupations, n_genres=n_genres, embed_dim=embed_dim)
        self.item_tower = ItemTower(n_items, n_genres=n_genres, embed_dim=embed_dim)

    def forward(self, user_batch: dict, item_batch: dict) -> torch.Tensor:
        """
        Args:
            user_batch: dict of tensors matching UserTower.forward() signature
            item_batch: dict of tensors matching ItemTower.forward() signature

        Returns:
            (B,) dot-product scores (cosine similarity)
        """
        u_emb = self.user_tower(**user_batch)
        i_emb = self.item_tower(**item_batch)
        return (u_emb * i_emb).sum(dim=-1)  # dot product → scalar score
