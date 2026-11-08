"""Conversations (meta_db). Always scoped to the owning user."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ErrorCode
from app.db.models import Conversation


def get_or_create(db: Session, user_id: int, conversation_id: uuid.UUID | None,
                  *, title: str | None = None) -> Conversation:
    if conversation_id is not None:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user_id:
            raise AppError(ErrorCode.NOT_FOUND, "Conversation not found.")
        return conv
    conv = Conversation(id=uuid.uuid4(), user_id=user_id, title=(title or "")[:200] or None)
    db.add(conv)
    db.commit()
    return conv


def list_for_user(db: Session, user_id: int, limit: int = 50) -> list[Conversation]:
    stmt = (select(Conversation).where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc()).limit(limit))
    return list(db.execute(stmt).scalars())
