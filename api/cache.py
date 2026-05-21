"""
Cache layer — wraps Redis (local) or Upstash Redis (production).

Auto-detects which backend to use based on environment variables:
- If UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN are set → use upstash_redis
- Else if REDIS_URL is set → use the standard `redis` library
- Else → caching is disabled (all ops return None / no-op)

All public functions gracefully swallow connection errors so the API
continues working without cache when Redis is unavailable.
"""

import json
import os
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------
TTL_RECS = 3600       # rec:{user_id}:{top_n}   — 1 hour
TTL_SIMILAR = 86400   # sim:{movie_id}:{top_n}  — 24 hours
TTL_USER_FEATS = 43200  # user_feats:{user_id}  — 12 hours
TTL_ITEM_META = 86400   # item_meta:{movie_id}  — 24 hours

# ---------------------------------------------------------------------------
# Redis client singleton
# ---------------------------------------------------------------------------
_redis_client: Any = None
_backend: str = "none"   # "upstash" | "local" | "none"


def _init_client() -> None:
    """Lazily initialise the Redis client the first time it's needed."""
    global _redis_client, _backend

    if _redis_client is not None:
        return

    upstash_url = os.environ.get("UPSTASH_REDIS_URL")
    upstash_token = os.environ.get("UPSTASH_REDIS_TOKEN")
    redis_url = os.environ.get("REDIS_URL")

    # Priority 1: Upstash (production)
    if upstash_url and upstash_token:
        try:
            from upstash_redis import Redis as UpstashRedis  # type: ignore

            _redis_client = UpstashRedis(url=upstash_url, token=upstash_token)
            _backend = "upstash"
            logger.info(f"Cache backend: Upstash Redis ({upstash_url[:40]}…)")
            return
        except ImportError:
            logger.warning("upstash_redis package not installed — falling back to local")
        except Exception as exc:
            logger.warning(f"Failed to connect to Upstash Redis: {exc}")

    # Priority 2: Local Redis
    if redis_url:
        try:
            import redis as redis_lib  # type: ignore

            _redis_client = redis_lib.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=3
            )
            # Test the connection
            _redis_client.ping()
            _backend = "local"
            logger.info(f"Cache backend: Local Redis ({redis_url})")
            return
        except ImportError:
            logger.warning("redis package not installed — cache disabled")
        except Exception as exc:
            logger.warning(f"Failed to connect to local Redis: {exc}")
            _redis_client = None

    # No cache available
    _backend = "none"
    logger.warning("No Redis configured — caching disabled")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached(key: str) -> Any | None:
    """Retrieve a JSON-serialised value from the cache. Returns None on miss."""
    _init_client()
    if _redis_client is None:
        return None

    try:
        if _backend == "upstash":
            raw = _redis_client.get(key)
        else:
            raw = _redis_client.get(key)

        if raw is None:
            return None

        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Value exists but is not valid JSON — treat as miss
        return None
    except Exception as exc:
        logger.warning(f"Cache GET failed for '{key}': {exc}")
        return None


def set_cached(key: str, value: Any, ttl: int) -> None:
    """Serialise *value* to JSON and store it with a TTL (seconds)."""
    _init_client()
    if _redis_client is None:
        return

    try:
        payload = json.dumps(value)
        if _backend == "upstash":
            _redis_client.set(key, payload, ex=ttl)
        else:
            _redis_client.setex(key, ttl, payload)
    except Exception as exc:
        logger.warning(f"Cache SET failed for '{key}': {exc}")


def invalidate_user_cache(user_id: int) -> None:
    """Delete all cached recommendations for a user.

    Scans for keys matching ``rec:{user_id}:*`` and deletes them.
    Also removes the cached user features key.
    """
    _init_client()
    if _redis_client is None:
        return

    try:
        pattern = f"rec:{user_id}:*"
        user_feat_key = f"user_feats:{user_id}"

        if _backend == "upstash":
            # Upstash supports SCAN-based deletion through its API
            cursor = 0
            keys_to_delete: list[str] = []
            while True:
                cursor, keys = _redis_client.scan(cursor, match=pattern, count=100)
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break
            if keys_to_delete:
                _redis_client.delete(*keys_to_delete)
            _redis_client.delete(user_feat_key)
        else:
            # Local Redis
            keys_to_delete = []
            for key in _redis_client.scan_iter(match=pattern, count=100):
                keys_to_delete.append(key)
            if keys_to_delete:
                _redis_client.delete(*keys_to_delete)
            _redis_client.delete(user_feat_key)

        logger.debug(f"Invalidated cache for user {user_id} ({len(keys_to_delete)} rec keys)")
    except Exception as exc:
        logger.warning(f"Cache invalidation failed for user {user_id}: {exc}")


def delete_key(key: str) -> None:
    """Delete a single cache key."""
    _init_client()
    if _redis_client is None:
        return

    try:
        _redis_client.delete(key)
    except Exception as exc:
        logger.warning(f"Cache DELETE failed for '{key}': {exc}")


def is_available() -> bool:
    """Check if the cache backend is available."""
    _init_client()
    if _redis_client is None:
        return False

    try:
        if _backend == "upstash":
            _redis_client.ping()
        else:
            _redis_client.ping()
        return True
    except Exception:
        return False
