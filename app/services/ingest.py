"""인제스트 오케스트레이션 서비스 - 경찰청 API → 임베딩 → Qdrant 파이프라인"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

import httpx

from app.models.ingest import IngestJobState, IngestRequest, IngestStatus
from app.services.embedding import EmbeddingService
from app.services.police_client import PoliceAPIClient
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class IngestService:
    """
    경찰청 API에서 데이터를 가져와 CLIP 임베딩 후 Qdrant에 저장.

    실행 흐름:
    1. PoliceAPIClient.fetch_all() → AsyncGenerator[PoliceAPIItem]
    2. 각 항목마다:
       a. text_vec = CLIP(물품명 + 게시제목 + 분류명)
       b. image_vec = CLIP(이미지 URL 다운로드) | None
    3. 100건 배치마다 Qdrant upsert
    """

    def __init__(
        self,
        police_client: PoliceAPIClient,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        http_client: httpx.AsyncClient,
    ):
        self._police = police_client
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._http = http_client

        # job_id → IngestJobState (인메모리 추적)
        self._jobs: dict[str, IngestJobState] = {}

    def create_job(self, request: IngestRequest) -> IngestJobState:
        """새 인제스트 작업 생성"""
        job_id = str(uuid.uuid4())
        job = IngestJobState(job_id=job_id, request=request)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> IngestJobState | None:
        return self._jobs.get(job_id)

    async def run_job(self, job: IngestJobState) -> None:
        """백그라운드로 실행될 인제스트 작업 (asyncio.create_task로 호출)"""
        job.status = IngestStatus.RUNNING
        job.started_at = datetime.now(tz=timezone.utc)
        req = job.request

        try:
            await self._vector_store.ensure_collection()

            # 이미지 동시 다운로드 세마포어 (과부하 방지)
            image_semaphore = asyncio.Semaphore(10)
            batch: list = []  # (item, text_vec, image_vec | None)

            async for item in self._police.fetch_all(
                start_ymd=req.start_ymd,
                end_ymd=req.end_ymd,
                num_of_rows=req.num_of_rows,
            ):
                job.items_processed += 1

                # 텍스트 임베딩
                text_for_embed = item.build_text_for_embedding()
                text_vec = await self._embedding.encode_text(text_for_embed)

                # 이미지 임베딩 (있을 때만)
                image_vec = None
                if item.fdFilePathImg:
                    async with image_semaphore:
                        image_vec = await self._embedding.encode_image_from_url(
                            item.fdFilePathImg, self._http
                        )
                    if image_vec is not None:
                        job.items_with_images += 1

                batch.append((item, text_vec, image_vec))

                # 100건마다 배치 upsert
                if len(batch) >= _BATCH_SIZE:
                    upserted = await self._vector_store.upsert_batch(batch)
                    job.items_upserted += upserted
                    logger.info(
                        "[Job %s] 진행: %d건 처리됨 (이미지: %d)",
                        job.job_id[:8],
                        job.items_processed,
                        job.items_with_images,
                    )
                    batch.clear()

            # 남은 배치 처리
            if batch:
                upserted = await self._vector_store.upsert_batch(batch)
                job.items_upserted += upserted

            job.status = IngestStatus.COMPLETED
            job.completed_at = datetime.now(tz=timezone.utc)
            logger.info(
                "[Job %s] 완료: 총 %d건 upserted (이미지: %d건)",
                job.job_id[:8],
                job.items_upserted,
                job.items_with_images,
            )

        except Exception as e:
            job.status = IngestStatus.FAILED
            job.errors.append(str(e))
            job.completed_at = datetime.now(tz=timezone.utc)
            logger.exception("[Job %s] 실패: %s", job.job_id[:8], e)
