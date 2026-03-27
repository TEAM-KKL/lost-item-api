"""분실물 검색 엔드포인트"""

import time
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agent.agent import run_search_agent
from app.api.deps import get_embedding_service, get_vector_store
from app.models.search import (
    CombinedSearchRequest,
    SearchResponse,
    TextSearchRequest,
)
from app.services.embedding import EmbeddingService  # noqa: F401 (Depends 타입 힌트용)
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


# ──────────────────────────── 텍스트 검색 ────────────────────────────

@router.post("/text", response_model=SearchResponse, summary="텍스트로 분실물 검색")
async def search_by_text(
    request: TextSearchRequest,
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> SearchResponse:
    """
    텍스트로 분실물을 검색합니다.

    - **use_agent=true**: GPT-4o 에이전트가 쿼리를 분석하고 최적화된 검색 수행
    - **use_agent=false**: CLIP 벡터로 직접 검색 (빠름)
    """
    start = time.perf_counter()

    if request.use_agent:
        try:
            agent_result = await run_search_agent(
                user_query=request.query,
                embedding_service=embedding,
                vector_store=vector_store,
                top_k=request.top_k,
                filter_category=request.filter_category,
                filter_date_from=request.filter_date_from,
                filter_date_to=request.filter_date_to,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return SearchResponse(
                items=agent_result.items,
                total=len(agent_result.items),
                agent_reasoning=agent_result.reasoning,
                query_metadata=agent_result.metadata,
                search_time_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            logger.warning("에이전트 실행 실패, 원시 검색으로 폴백: %s", e)
            # 에이전트 실패 시 원시 검색으로 폴백

    # 원시 벡터 검색
    query_vec = await embedding.encode_text(request.query)
    results = await vector_store.search_by_text(
        query_vec=query_vec,
        top_k=request.top_k,
        filter_category=request.filter_category,
        filter_date_from=request.filter_date_from,
        filter_date_to=request.filter_date_to,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        search_time_ms=round(elapsed_ms, 2),
    )


# ──────────────────────────── 이미지 검색 ────────────────────────────

@router.post("/image", response_model=SearchResponse, summary="이미지로 분실물 검색")
async def search_by_image(
    file: UploadFile = File(..., description="검색할 이미지 파일 (jpg, png 등)"),
    top_k: int = Form(default=10, ge=1, le=50),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> SearchResponse:
    """
    이미지를 업로드하여 시각적으로 유사한 습득물을 검색합니다.

    - CLIP 모델로 이미지와 텍스트가 동일한 벡터 공간에 임베딩되어 있어
      이미지로 텍스트 설명도 함께 검색됩니다.
    """
    start = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일(jpg, png 등)만 업로드 가능합니다")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    try:
        image_vec = await embedding.encode_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {e}") from e

    results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        search_time_ms=round(elapsed_ms, 2),
    )


# ──────────────────────────── 복합 검색 ────────────────────────────

@router.post("/combined", response_model=SearchResponse, summary="텍스트 + 이미지 복합 검색")
async def search_combined(
    query: str | None = Form(default=None, description="분실물 텍스트 설명"),
    file: UploadFile | None = File(default=None, description="분실물 이미지 (선택)"),
    top_k: int = Form(default=10, ge=1, le=50),
    text_weight: float = Form(default=0.5, ge=0.0, le=1.0, description="텍스트 가중치 (이미지=1-weight)"),
    use_agent: bool = Form(default=True),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> SearchResponse:
    """
    텍스트와 이미지를 함께 사용해 분실물을 검색합니다.

    - 텍스트만: 텍스트 벡터 검색
    - 이미지만: 이미지 벡터 검색
    - 둘 다: 가중 평균 벡터로 검색
    """
    start = time.perf_counter()

    if not query and not file:
        raise HTTPException(status_code=400, detail="텍스트 또는 이미지 중 하나는 입력해야 합니다")

    text_vec: list[float] | None = None
    image_vec: list[float] | None = None

    if query:
        text_vec = await embedding.encode_text(query)

    if file and file.filename:
        image_bytes = await file.read()
        if image_bytes:
            try:
                image_vec = await embedding.encode_image_from_bytes(image_bytes)
            except Exception as e:
                logger.warning("이미지 처리 실패 (텍스트 검색으로 진행): %s", e)

    # 검색 벡터 결정
    if text_vec is not None and image_vec is not None:
        # 텍스트→text_vec, 이미지→image_vec 각각 검색 후 RRF 병합
        results = await vector_store.search_combined(
            text_vec=text_vec,
            image_vec=image_vec,
            top_k=top_k,
        )
        matched_mode = "combined"
    elif image_vec is not None:
        results = await vector_store.search_by_image(image_vec, top_k=top_k)
        matched_mode = "image"
    else:
        # text_vec만 있는 경우 (에이전트 옵션)
        if use_agent and query:
            try:
                agent_result = await run_search_agent(
                    user_query=query,
                    embedding_service=embedding,
                    vector_store=vector_store,
                    top_k=top_k,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                return SearchResponse(
                    items=agent_result.items,
                    total=len(agent_result.items),
                    agent_reasoning=agent_result.reasoning,
                    query_metadata=agent_result.metadata,
                    search_time_ms=round(elapsed_ms, 2),
                )
            except Exception as e:
                logger.warning("에이전트 실패, 원시 검색 폴백: %s", e)

        results = await vector_store.search_by_text(text_vec, top_k=top_k)  # type: ignore[arg-type]
        matched_mode = "text"

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        search_time_ms=round(elapsed_ms, 2),
    )
