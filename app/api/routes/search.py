"""Search API routes."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agent.agent import run_search_agent
from app.api.deps import (
    get_embedding_service,
    get_search_session_service,
    get_vector_store,
)
from app.models.search import SearchResponse, TextSearchRequest
from app.services.embedding import EmbeddingService
from app.services.search_session import (
    SearchSessionService,
    SessionContext,
    SessionFilters,
    SessionMessage,
)
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


def _resolve_session_id(
    session_service: SearchSessionService,
    requested_session_id: str | None,
    *,
    enable_session: bool,
) -> str | None:
    if not enable_session:
        return None
    return requested_session_id or session_service.generate_session_id()


def _merge_filters(request_filters: SessionFilters, stored_filters: SessionFilters) -> SessionFilters:
    return SessionFilters(
        filter_category=request_filters.filter_category or stored_filters.filter_category,
        filter_date_from=request_filters.filter_date_from or stored_filters.filter_date_from,
        filter_date_to=request_filters.filter_date_to or stored_filters.filter_date_to,
    )


def _build_result_summary(
    *,
    query: str,
    total: int,
    filters: SessionFilters,
    agent_reasoning: str | None = None,
) -> str:
    parts = [f"'{query}' 검색 결과 {total}건"]
    if filters.filter_category:
        parts.append(f"카테고리={filters.filter_category}")
    if filters.filter_date_from or filters.filter_date_to:
        parts.append(f"기간={filters.filter_date_from or '-'}~{filters.filter_date_to or '-'}")
    if agent_reasoning:
        parts.append(f"요약={agent_reasoning}")
    return " | ".join(parts)


async def _load_session_context(
    session_service: SearchSessionService,
    session_id: str | None,
) -> SessionContext | None:
    if not session_id:
        return None
    try:
        return await session_service.load_session_context(session_id)
    except Exception as exc:
        logger.warning("세션 로드 실패, 단발 검색으로 진행: %s", exc)
        return None


async def _persist_session_turn(
    session_service: SearchSessionService,
    session_id: str | None,
    *,
    user_query: str,
    assistant_summary: str,
    filters: SessionFilters,
    existing_summary: str = "",
) -> None:
    if not session_id:
        return

    try:
        await session_service.append_session_messages(
            session_id,
            [
                SessionMessage(role="user", content=user_query),
                SessionMessage(role="assistant", content=assistant_summary),
            ],
        )
        await session_service.update_session_summary(session_id, existing_summary, filters)
    except Exception as exc:
        logger.warning("세션 저장 실패, 검색 결과만 반환: %s", exc)


@router.post("/text", response_model=SearchResponse, summary="텍스트로 분실물 검색")
async def search_by_text(
    request: TextSearchRequest,
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    session_service: SearchSessionService = Depends(get_search_session_service),
) -> SearchResponse:
    start = time.perf_counter()
    session_id = _resolve_session_id(session_service, request.session_id, enable_session=True)
    session_context = await _load_session_context(session_service, session_id)

    request_filters = SessionFilters(
        filter_category=request.filter_category,
        filter_date_from=request.filter_date_from,
        filter_date_to=request.filter_date_to,
    )
    effective_filters = _merge_filters(
        request_filters,
        session_context.last_filters if session_context else SessionFilters(),
    )

    agent_reasoning: str | None = None
    query_metadata = None

    if request.use_agent:
        try:
            agent_result = await run_search_agent(
                user_query=request.query,
                embedding_service=embedding,
                vector_store=vector_store,
                top_k=request.top_k,
                filter_category=effective_filters.filter_category,
                filter_date_from=effective_filters.filter_date_from,
                filter_date_to=effective_filters.filter_date_to,
                session_summary=session_context.summary if session_context else "",
                recent_messages=session_context.recent_messages if session_context else None,
                inherited_filters=session_context.last_filters if session_context else None,
            )
            results = agent_result.items
            agent_reasoning = agent_result.reasoning
            query_metadata = agent_result.metadata
            if not effective_filters.filter_category and agent_result.metadata.item_type:
                effective_filters.filter_category = agent_result.metadata.item_type
        except Exception as exc:
            logger.warning("에이전트 실행 실패, 원시 검색으로 폴백: %s", exc)
            query_vec = await embedding.encode_text(request.query)
            results = await vector_store.search_by_text(
                query_vec=query_vec,
                top_k=request.top_k,
                filter_category=effective_filters.filter_category,
                filter_date_from=effective_filters.filter_date_from,
                filter_date_to=effective_filters.filter_date_to,
            )
    else:
        query_vec = await embedding.encode_text(request.query)
        results = await vector_store.search_by_text(
            query_vec=query_vec,
            top_k=request.top_k,
            filter_category=effective_filters.filter_category,
            filter_date_from=effective_filters.filter_date_from,
            filter_date_to=effective_filters.filter_date_to,
        )

    assistant_summary = _build_result_summary(
        query=request.query,
        total=len(results),
        filters=effective_filters,
        agent_reasoning=agent_reasoning,
    )
    await _persist_session_turn(
        session_service,
        session_id,
        user_query=request.query,
        assistant_summary=assistant_summary,
        filters=effective_filters,
        existing_summary=session_context.summary if session_context else "",
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=session_id,
        agent_reasoning=agent_reasoning,
        query_metadata=query_metadata,
        search_time_ms=round(elapsed_ms, 2),
    )


@router.post("/image", response_model=SearchResponse, summary="이미지로 분실물 검색")
async def search_by_image(
    file: UploadFile = File(..., description="검색할 이미지 파일"),
    top_k: int = Form(default=10, ge=1, le=50),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> SearchResponse:
    start = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        image_vec = await embedding.encode_image_from_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {exc}") from exc

    results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=None,
        search_time_ms=round(elapsed_ms, 2),
    )


@router.post("/combined", response_model=SearchResponse, summary="텍스트 + 이미지 복합 검색")
async def search_combined(
    query: str | None = Form(default=None, description="분실물 설명 텍스트"),
    file: UploadFile | None = File(default=None, description="분실물 이미지"),
    top_k: int = Form(default=10, ge=1, le=50),
    text_weight: float = Form(default=0.5, ge=0.0, le=1.0, description="텍스트 가중치"),
    use_agent: bool = Form(default=True),
    session_id: str | None = Form(default=None),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    session_service: SearchSessionService = Depends(get_search_session_service),
) -> SearchResponse:
    del text_weight  # 현재는 RRF 기반 병합만 지원
    start = time.perf_counter()

    if not query and not file:
        raise HTTPException(status_code=400, detail="텍스트 또는 이미지를 하나 이상 입력해야 합니다.")

    resolved_session_id = _resolve_session_id(
        session_service,
        session_id,
        enable_session=bool(query),
    )
    session_context = await _load_session_context(session_service, resolved_session_id)

    text_vec: list[float] | None = None
    image_vec: list[float] | None = None
    agent_reasoning: str | None = None
    query_metadata = None

    if query:
        text_vec = await embedding.encode_text(query)

    if file and file.filename:
        image_bytes = await file.read()
        if image_bytes:
            try:
                image_vec = await embedding.encode_image_from_bytes(image_bytes)
            except Exception as exc:
                logger.warning("이미지 처리 실패, 텍스트 검색으로 계속 진행: %s", exc)

    if text_vec is not None and image_vec is not None:
        results = await vector_store.search_combined(
            text_vec=text_vec,
            image_vec=image_vec,
            top_k=top_k,
        )
    elif image_vec is not None:
        results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
    else:
        assert query is not None
        if use_agent:
            try:
                agent_result = await run_search_agent(
                    user_query=query,
                    embedding_service=embedding,
                    vector_store=vector_store,
                    top_k=top_k,
                    session_summary=session_context.summary if session_context else "",
                    recent_messages=session_context.recent_messages if session_context else None,
                    inherited_filters=session_context.last_filters if session_context else None,
                )
                results = agent_result.items
                agent_reasoning = agent_result.reasoning
                query_metadata = agent_result.metadata
            except Exception as exc:
                logger.warning("에이전트 실행 실패, 원시 검색으로 폴백: %s", exc)
                results = await vector_store.search_by_text(text_vec, top_k=top_k)
        else:
            results = await vector_store.search_by_text(text_vec, top_k=top_k)

    if query:
        await _persist_session_turn(
            session_service,
            resolved_session_id,
            user_query=query,
            assistant_summary=_build_result_summary(
                query=query,
                total=len(results),
                filters=session_context.last_filters if session_context else SessionFilters(),
                agent_reasoning=agent_reasoning,
            ),
            filters=session_context.last_filters if session_context else SessionFilters(),
            existing_summary=session_context.summary if session_context else "",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=resolved_session_id,
        agent_reasoning=agent_reasoning,
        query_metadata=query_metadata,
        search_time_ms=round(elapsed_ms, 2),
    )
