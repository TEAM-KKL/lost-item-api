"""MongoDB-backed search session storage."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_RECENT_MESSAGE_LIMIT = 6
_MESSAGE_PREVIEW_LIMIT = 140
_SUMMARY_MAX_CHARS = 1200


@dataclass
class SessionFilters:
    filter_category: str | None = None
    filter_date_from: str | None = None
    filter_date_to: str | None = None


@dataclass
class SessionMessage:
    role: str
    content: str
    created_at: datetime | None = None


@dataclass
class SessionContext:
    session_id: str
    summary: str = ""
    recent_messages: list[SessionMessage] = field(default_factory=list)
    last_filters: SessionFilters = field(default_factory=SessionFilters)


def summarize_messages(
    existing_summary: str,
    messages: list[SessionMessage],
    *,
    max_chars: int = _SUMMARY_MAX_CHARS,
) -> str:
    """Build a compact Korean summary from older conversation turns."""
    snippets: list[str] = []
    if existing_summary.strip():
        snippets.append(existing_summary.strip())

    for message in messages:
        content = " ".join(message.content.split())
        if not content:
            continue
        if len(content) > _MESSAGE_PREVIEW_LIMIT:
            content = f"{content[:_MESSAGE_PREVIEW_LIMIT - 3]}..."
        role_label = "사용자" if message.role == "user" else "어시스턴트"
        snippets.append(f"{role_label}: {content}")

    merged = " | ".join(snippets)
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]


class SearchSessionService:
    """Persist search chat sessions in MongoDB."""

    DB_NAME = "search"
    SESSION_COLLECTION = "search_sessions"
    MESSAGE_COLLECTION = "search_messages"

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017"):
        self._client = AsyncIOMotorClient(mongo_uri)
        self._db: AsyncIOMotorDatabase = self._client[self.DB_NAME]
        self._sessions: AsyncIOMotorCollection = self._db[self.SESSION_COLLECTION]
        self._messages: AsyncIOMotorCollection = self._db[self.MESSAGE_COLLECTION]

    async def ensure_indexes(self) -> None:
        await self._sessions.create_index("session_id", unique=True)
        await self._sessions.create_index("updated_at")
        await self._messages.create_index([("session_id", 1), ("created_at", 1)])

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    async def load_session_context(self, session_id: str) -> SessionContext:
        session_doc = await self._sessions.find_one({"session_id": session_id}) or {}
        recent_docs = await (
            self._messages
            .find({"session_id": session_id})
            .sort("created_at", -1)
            .to_list(length=_RECENT_MESSAGE_LIMIT)
        )
        recent_docs.reverse()
        recent_messages = [
            SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                created_at=doc.get("created_at"),
            )
            for doc in recent_docs
        ]
        return SessionContext(
            session_id=session_id,
            summary=session_doc.get("summary", ""),
            recent_messages=recent_messages,
            last_filters=self._filters_from_doc(session_doc.get("last_filters")),
        )

    async def append_session_messages(self, session_id: str, messages: list[SessionMessage]) -> None:
        now = datetime.now(timezone.utc)
        docs = [
            {
                "session_id": session_id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at or now,
            }
            for message in messages
            if message.content.strip()
        ]
        await self._sessions.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "summary": "",
                    "last_filters": {},
                    "updated_at": now,
                },
            },
            upsert=True,
        )
        if docs:
            await self._messages.insert_many(docs)

    async def update_session_summary(
        self,
        session_id: str,
        summary: str,
        last_filters: SessionFilters,
    ) -> None:
        now = datetime.now(timezone.utc)
        session_doc = await self._sessions.find_one({"session_id": session_id}) or {}
        all_docs = await (
            self._messages
            .find({"session_id": session_id})
            .sort("created_at", 1)
            .to_list(length=1000)
        )

        merged_summary = summary.strip() or session_doc.get("summary", "")
        if len(all_docs) > _RECENT_MESSAGE_LIMIT:
            old_docs = all_docs[:-_RECENT_MESSAGE_LIMIT]
            old_messages = [
                SessionMessage(
                    role=doc.get("role", "user"),
                    content=doc.get("content", ""),
                    created_at=doc.get("created_at"),
                )
                for doc in old_docs
            ]
            merged_summary = summarize_messages(
                session_doc.get("summary", "") or merged_summary,
                old_messages,
            )
            old_ids = [doc["_id"] for doc in old_docs if "_id" in doc]
            if old_ids:
                await self._messages.delete_many({"_id": {"$in": old_ids}})

        await self._sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "summary": merged_summary,
                    "last_filters": self._filters_to_doc(last_filters),
                    "updated_at": now,
                },
                "$setOnInsert": {"session_id": session_id},
            },
            upsert=True,
        )

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _filters_to_doc(filters: SessionFilters) -> dict[str, str]:
        doc: dict[str, str] = {}
        if filters.filter_category:
            doc["filter_category"] = filters.filter_category
        if filters.filter_date_from:
            doc["filter_date_from"] = filters.filter_date_from
        if filters.filter_date_to:
            doc["filter_date_to"] = filters.filter_date_to
        return doc

    @staticmethod
    def _filters_from_doc(doc: Any) -> SessionFilters:
        if not isinstance(doc, dict):
            return SessionFilters()
        return SessionFilters(
            filter_category=doc.get("filter_category"),
            filter_date_from=doc.get("filter_date_from"),
            filter_date_to=doc.get("filter_date_to"),
        )
