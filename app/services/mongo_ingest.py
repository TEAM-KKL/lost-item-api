"""MongoDB → CLIP 임베딩 → Qdrant 인제스트 서비스"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.models.ingest import IngestJobState, IngestRequest, IngestStatus
from app.models.mongo_item import MongoLostItem
from app.services.embedding import EmbeddingService
from app.services.mongo_client import MongoLostItemClient
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50   # 로컬 이미지 포함 배치 크기


class MongoIngestService:
    """
    MongoDB crawl.lost_items → CLIP 임베딩 → Qdrant 파이프라인.

    이미지는 로컬 downloads 디렉터리에서 직접 읽어 임베딩합니다.
    (네트워크 다운로드 없음 → 빠름)
    """

    def __init__(
        self,
        mongo_client: MongoLostItemClient,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        downloads_dir: str,
    ):
        self._mongo = mongo_client
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._downloads_dir = downloads_dir
        self._jobs: dict[str, IngestJobState] = {}
        self._job_params: dict[str, dict] = {}

    def create_job(self, skip: int = 0, limit: int = 0) -> IngestJobState:
        """새 인제스트 작업 생성"""
        job_id = str(uuid.uuid4())
        req = IngestRequest(start_ymd="2020-01-01", end_ymd="2030-12-31")
        job = IngestJobState(job_id=job_id, request=req)
        self._jobs[job_id] = job
        self._job_params[job_id] = {"skip": skip, "limit": limit}
        return job

    def get_job(self, job_id: str) -> IngestJobState | None:
        return self._jobs.get(job_id)

    async def run_job(self, job: IngestJobState) -> None:
        """BackgroundTasks로 실행될 인제스트 작업"""
        job.status = IngestStatus.RUNNING
        job.started_at = datetime.now(tz=timezone.utc)
        params = self._job_params.get(job.job_id, {})
        skip = params.get("skip", 0)
        limit = params.get("limit", 0)

        try:
            await self._vector_store.ensure_collection()

            total = await self._mongo.count()
            effective_total = (total - skip) if limit == 0 else min(limit, total - skip)
            job.total_pages = max(1, effective_total // _BATCH_SIZE + 1)
            logger.info(
                "MongoDB 인제스트 시작: 총 %d건 → 처리 대상 %d건 (skip=%d)",
                total, effective_total, skip,
            )

            img_sem = asyncio.Semaphore(8)
            batch: list[tuple[MongoLostItem, list[float], list[float] | None]] = []

            async for item in self._mongo.iter_all(
                batch_size=_BATCH_SIZE * 2,
                skip=skip,
                limit=limit,
            ):
                job.items_processed += 1

                # 텍스트 임베딩
                text_vec = await self._embedding.encode_text(
                    item.build_text_for_embedding()
                )

                # 로컬 이미지 임베딩
                image_vec: list[float] | None = None
                local_path = item.get_local_image_path(self._downloads_dir)
                if local_path:
                    async with img_sem:
                        try:
                            img_bytes = await asyncio.to_thread(local_path.read_bytes)
                            image_vec = await self._embedding.encode_image_from_bytes(img_bytes)
                            job.items_with_images += 1
                        except Exception as e:
                            logger.debug("이미지 임베딩 실패 (%s): %s", local_path.name, e)

                batch.append((item, text_vec, image_vec))

                if len(batch) >= _BATCH_SIZE:
                    job.items_upserted += await self._vector_store.upsert_mongo_batch(batch)
                    job.pages_fetched += 1
                    logger.info(
                        "[Mongo Job %s] %d건 처리됨 (이미지: %d)",
                        job.job_id[:8], job.items_processed, job.items_with_images,
                    )
                    batch.clear()

            if batch:
                job.items_upserted += await self._vector_store.upsert_mongo_batch(batch)
                job.pages_fetched += 1

            job.status = IngestStatus.COMPLETED
            job.completed_at = datetime.now(tz=timezone.utc)
            logger.info(
                "[Mongo Job %s] 완료: %d건 upserted (이미지: %d건)",
                job.job_id[:8], job.items_upserted, job.items_with_images,
            )

        except Exception as e:
            job.status = IngestStatus.FAILED
            job.errors.append(str(e))
            job.completed_at = datetime.now(tz=timezone.utc)
            logger.exception("[Mongo Job %s] 실패: %s", job.job_id[:8], e)
