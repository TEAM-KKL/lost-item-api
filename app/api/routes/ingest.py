"""데이터 수집(인제스트) 엔드포인트"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi import status as http_status

from app.api.deps import get_ingest_service
from app.models.ingest import (
    IngestRequest,
    IngestResponse,
    IngestStatus,
    IngestStatusResponse,
)
from app.services.ingest import IngestService

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


@router.post(
    "/sync",
    response_model=IngestResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="경찰청 습득물 데이터 수집 시작",
)
async def start_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    """
    경찰청 공공데이터 API에서 습득물 데이터를 가져와 Qdrant에 임베딩 저장합니다.

    - 백그라운드 작업으로 실행 (즉시 job_id 반환)
    - `/ingest/status/{job_id}` 로 진행 상황 확인
    - 날짜 범위가 넓을수록 시간이 오래 걸립니다 (이미지 다운로드 포함)

    **예시 요청**:
    ```json
    {
      "start_ymd": "2024-01-01",
      "end_ymd": "2024-03-31",
      "num_of_rows": 100
    }
    ```
    """
    job = ingest_service.create_job(request)
    background_tasks.add_task(ingest_service.run_job, job)

    logger.info("인제스트 작업 시작: job_id=%s, 기간=%s~%s", job.job_id, request.start_ymd, request.end_ymd)
    return IngestResponse(
        job_id=job.job_id,
        status=IngestStatus.RUNNING,
        message=f"데이터 수집 작업을 시작했습니다 (job_id: {job.job_id[:8]}...)",
    )


@router.get(
    "/status/{job_id}",
    response_model=IngestStatusResponse,
    summary="인제스트 작업 진행 상황 조회",
)
async def get_ingest_status(
    job_id: str,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestStatusResponse:
    """인제스트 작업의 현재 진행 상황을 반환합니다."""
    job = ingest_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}'를 찾을 수 없습니다")

    return IngestStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        pages_fetched=job.pages_fetched,
        total_pages=job.total_pages,
        items_processed=job.items_processed,
        items_upserted=job.items_upserted,
        items_with_images=job.items_with_images,
        errors=job.errors,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
