"""헬스체크 엔드포인트"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_vector_store
from app.services.vector_store import VectorStoreService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


class QdrantHealthResponse(BaseModel):
    status: str
    collection: str
    vectors_count: int | None = None
    points_count: int | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """앱 상태 확인"""
    return HealthResponse(status="ok", version="1.0.0")


@router.get("/health/qdrant", response_model=QdrantHealthResponse)
async def qdrant_health(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> QdrantHealthResponse:
    """Qdrant 연결 및 컬렉션 상태 확인"""
    try:
        info = await vector_store.get_collection_info()
        return QdrantHealthResponse(
            status="ok",
            collection=vector_store.COLLECTION,
            vectors_count=info.get("vectors_count"),
            points_count=info.get("points_count"),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant 연결 실패: {e}") from e
