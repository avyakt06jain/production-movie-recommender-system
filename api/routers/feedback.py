"""
POST /api/v1/feedback

Ingests user interaction events (click, watch, skip, rate) into the
``feedback_events`` PostgreSQL table and invalidates the Redis cache
for the affected user so the next recommendation request is fresh.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache import invalidate_user_cache
from api.database import get_db

router = APIRouter(tags=["feedback"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = {"click", "watch", "skip", "rate"}


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user")
    movie_id: int = Field(..., description="ID of the movie")
    event_type: str = Field(
        ...,
        description="Type of interaction: click, watch, skip, or rate",
    )
    session_id: str = Field(..., description="Client session identifier")
    rating: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="Rating value (required for 'rate' events)"
    )


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    event_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    # Validate event type
    if body.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid event_type '{body.event_type}'. Must be one of: {sorted(VALID_EVENT_TYPES)}",
        )

    # Rating is required for 'rate' events
    if body.event_type == "rate" and body.rating is None:
        raise HTTPException(
            status_code=422,
            detail="A 'rating' value (1.0-5.0) is required for 'rate' events.",
        )

    # Insert into feedback_events using raw SQL to stay compatible with
    # whatever DDL was applied (avoids needing an ORM model declaration here).
    insert_sql = text("""
        INSERT INTO feedback_events (user_id, movie_id, event_type, rating, session_id, timestamp)
        VALUES (:user_id, :movie_id, :event_type, :rating, :session_id, :ts)
        RETURNING event_id
    """)

    try:
        result = await db.execute(
            insert_sql,
            {
                "user_id": body.user_id,
                "movie_id": body.movie_id,
                "event_type": body.event_type,
                "rating": body.rating,
                "session_id": body.session_id,
                "ts": datetime.now(timezone.utc),
            },
        )
        row = result.fetchone()
        event_id = row[0] if row else None
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.error(f"Failed to insert feedback event: {exc}")
        raise HTTPException(status_code=500, detail="Failed to store feedback event.")

    # Invalidate cached recommendations for this user
    try:
        invalidate_user_cache(body.user_id)
    except Exception as exc:
        # Non-fatal — log and continue
        logger.warning(f"Cache invalidation failed for user {body.user_id}: {exc}")

    logger.info(
        f"Feedback recorded: user={body.user_id} movie={body.movie_id} "
        f"type={body.event_type} event_id={event_id}"
    )

    return FeedbackResponse(
        success=True,
        message="Feedback event recorded successfully.",
        event_id=event_id,
    )
