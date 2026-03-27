"""Motor(async) MongoDB 클라이언트 - crawl.lost_items 조회"""

import logging
from typing import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.models.mongo_item import MongoLostItem

logger = logging.getLogger(__name__)


class MongoLostItemClient:
    """
    crawl.lost_items 컬렉션을 비동기로 조회하는 클라이언트.

    - iter_all(): 전체 문서를 배치 단위로 스트리밍
    - count(): 전체 문서 수
    """

    DB_NAME = "crawl"
    COLLECTION_NAME = "lost_items"

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017"):
        self._client = AsyncIOMotorClient(mongo_uri)
        self._db: AsyncIOMotorDatabase = self._client[self.DB_NAME]
        self._col = self._db[self.COLLECTION_NAME]

    async def count(self) -> int:
        return await self._col.count_documents({})

    async def iter_all(
        self,
        batch_size: int = 200,
        skip: int = 0,
        limit: int = 0,
    ) -> AsyncGenerator[MongoLostItem, None]:
        """
        전체 문서를 AsyncGenerator로 스트리밍.
        메모리에 전체 로드 없이 배치 처리 가능.
        """
        cursor = self._col.find({}).skip(skip).batch_size(batch_size)
        if limit > 0:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            try:
                yield MongoLostItem(**doc)
            except Exception as e:
                logger.warning("MongoDB 문서 파싱 실패: %s | doc._id=%s", e, doc.get("_id"))

    async def close(self) -> None:
        self._client.close()
