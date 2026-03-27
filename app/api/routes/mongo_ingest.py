"""MongoDB 기반 인제스트 엔드포인트"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel, Field

from app.api.deps import get_vector_store
from app.models.ingest import IngestResponse, IngestStatus, IngestStatusResponse
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/ingest/mongo", tags=["ingest-mongo"])
logger = logging.getLogger(__name__)


class MongoIngestRequest(BaseModel):
    skip: int = Field(default=0, ge=0, description="건너뛸 문서 수 (재개 시 활용)")
    limit: int = Field(default=0, ge=0, description="처리할 최대 문서 수 (0=전체)")


def get_mongo_ingest_service(request_obj):
    from fastapi import Request
    req: Request = request_obj
    svc = getattr(req.app.state, "mongo_ingest_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="MongoDB 인제스트 서비스가 초기화되지 않았습니다")
    return svc


from fastapi import Request


@router.post(
    "",
    response_model=IngestResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="MongoDB 습득물 데이터 → Qdrant 임베딩 (백그라운드)",
)
async def start_mongo_ingest(
    body: MongoIngestRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> IngestResponse:
    """
    MongoDB `crawl.lost_items` 컬렉션의 데이터를 읽어
    CLIP 임베딩 후 Qdrant에 저장합니다.

    - 텍스트 임베딩: `search_text` 필드 활용
    - 이미지 임베딩: `downloads/` 로컬 폴더에서 직접 읽기
    - 백그라운드 실행 → `/ingest/mongo/status/{job_id}` 로 진행 상황 확인

    **예시**: 전체 12,837건 처리
    ```json
    { "skip": 0, "limit": 0 }
    ```
    **예시**: 중간부터 재개
    ```json
    { "skip": 5000, "limit": 0 }
    ```
    """
    svc = get_mongo_ingest_service(request)
    job = svc.create_job(skip=body.skip, limit=body.limit)
    background_tasks.add_task(svc.run_job, job)

    logger.info(
        "MongoDB 인제스트 시작: job_id=%s, skip=%d, limit=%d",
        job.job_id, body.skip, body.limit,
    )
    return IngestResponse(
        job_id=job.job_id,
        status=IngestStatus.RUNNING,
        message=f"MongoDB 임베딩 작업을 시작했습니다 (job_id: {job.job_id[:8]}...)",
    )


@router.get(
    "/status/{job_id}",
    response_model=IngestStatusResponse,
    summary="MongoDB 인제스트 진행 상황 조회",
)
async def get_mongo_ingest_status(job_id: str, request: Request) -> IngestStatusResponse:
    svc = get_mongo_ingest_service(request)
    job = svc.get_job(job_id)
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
