"""Search API routes."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.agent.agent import run_search_agent
from app.api.deps import (
    get_embedding_service,
    get_search_session_service,
    get_vector_store,
)
from app.models.search import (
    LostItemResult,
    RecentItemsResponse,
    SearchResponse,
    SessionHistoryResponse,
    SessionMessageResponse,
    TextSearchRequest,
)
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


def _filters_to_dict(filters: SessionFilters) -> dict[str, str | None]:
    return {
        "filter_category": filters.filter_category,
        "filter_date_from": filters.filter_date_from,
        "filter_date_to": filters.filter_date_to,
    }


def _build_assistant_message(
    *,
    query: str,
    results: list[LostItemResult],
    agent_reasoning: str | None = None,
) -> str:
    if agent_reasoning and agent_reasoning.strip():
        return agent_reasoning.strip()

    if not results:
        return f"'{query}'에 대한 검색 결과를 찾지 못했습니다. 다른 표현이나 장소, 날짜를 더 알려주시면 다시 찾아볼게요."

    headline = f"'{query}'와 관련된 결과 {len(results)}건을 찾았습니다."
    top_lines = []
    for index, item in enumerate(results[:3], start=1):
        top_lines.append(
            f"{index}. {item.fd_prdt_nm} / {item.dep_place} / {item.fd_ymd} / 유사도 {item.score:.3f}"
        )
    return headline + "\n" + "\n".join(top_lines)


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
    assistant_message: str,
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
                SessionMessage(role="assistant", content=assistant_message),
            ],
        )
        await session_service.update_session_summary(session_id, existing_summary, filters)
    except Exception as exc:
        logger.warning("세션 저장 실패, 검색 결과만 반환: %s", exc)


@router.get("/recent", response_model=RecentItemsResponse, summary="최근 등록된 분실물 목록 조회")
async def get_recent_items(
    limit: int = Query(default=20, ge=1, le=100, description="반환할 결과 수"),
    offset: int = Query(default=0, ge=0, description="건너뛸 결과 수 (페이지네이션)"),
    filter_category: str | None = Query(default=None, description="물품분류명 필터"),
    filter_date_from: str | None = Query(default=None, description="습득일 시작 (YYYY-MM-DD)"),
    filter_date_to: str | None = Query(default=None, description="습득일 종료 (YYYY-MM-DD)"),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> RecentItemsResponse:
    start = time.perf_counter()
    try:
        items, has_next = await vector_store.get_recent_items(
            limit=limit,
            offset=offset,
            filter_category=filter_category,
            filter_date_from=filter_date_from,
            filter_date_to=filter_date_to,
        )
    except Exception as exc:
        logger.error("최근 분실물 조회 실패: %s", exc)
        raise HTTPException(status_code=500, detail="최근 분실물 조회에 실패했습니다.") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    return RecentItemsResponse(
        items=items,
        total=len(items),
        has_next=has_next,
        search_time_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="세션 대화 기록 조회",
)
async def get_session_history(
    session_id: str,
    session_service: SearchSessionService = Depends(get_search_session_service),
) -> SessionHistoryResponse:
    try:
        session_context = await session_service.get_session_history(session_id)
    except Exception as exc:
        logger.warning("세션 기록 조회 실패: %s", exc)
        raise HTTPException(status_code=500, detail="세션 기록 조회에 실패했습니다.") from exc

    if session_context is None:
        raise HTTPException(status_code=404, detail="해당 session_id의 세션을 찾을 수 없습니다.")

    messages = [
        SessionMessageResponse(
            role=message.role,
            content=message.content,
            created_at=message.created_at.isoformat() if message.created_at else None,
        )
        for message in session_context.recent_messages
    ]
    return SessionHistoryResponse(
        session_id=session_context.session_id,
        summary=session_context.summary,
        last_filters=_filters_to_dict(session_context.last_filters),
        messages=messages,
    )


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
                session_id=session_id,
                session_context=session_context,
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

    assistant_message = _build_assistant_message(
        query=request.query,
        results=results,
        agent_reasoning=agent_reasoning,
    )
    await _persist_session_turn(
        session_service,
        session_id,
        user_query=request.query,
        assistant_message=assistant_message,
        filters=effective_filters,
        existing_summary=session_context.summary if session_context else "",
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=session_id,
        assistant_message=assistant_message,
        agent_reasoning=agent_reasoning,
        query_metadata=query_metadata,
        search_time_ms=round(elapsed_ms, 2),
    )


@router.post("/image", response_model=SearchResponse, summary="이미지로 분실물 검색")
async def search_by_image(
    file: UploadFile = File(..., description="검색할 이미지 파일"),
    top_k: int = Form(default=10, ge=1, le=50),
    use_agent: bool = Form(default=True),
    session_id: str | None = Form(default=None),
    embedding: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    session_service: SearchSessionService = Depends(get_search_session_service),
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

    agent_reasoning: str | None = None
    resolved_session_id = _resolve_session_id(session_service, session_id, enable_session=use_agent)
    session_context = await _load_session_context(session_service, resolved_session_id)

    if use_agent:
        try:
            agent_result = await run_search_agent(
                user_query="이미지로 유사한 분실물을 찾아주세요.",
                embedding_service=embedding,
                vector_store=vector_store,
                top_k=top_k,
                session_id=resolved_session_id,
                session_context=session_context,
                image_vec=image_vec,
            )
            results = agent_result.items
            agent_reasoning = agent_result.reasoning
        except Exception as exc:
            logger.warning("에이전트 실행 실패, 원시 검색으로 폴백: %s", exc)
            results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
    else:
        results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)

    assistant_message = _build_assistant_message(
        query="이미지 검색",
        results=results,
        agent_reasoning=agent_reasoning,
    )
    await _persist_session_turn(
        session_service,
        resolved_session_id,
        user_query="이미지로 유사한 분실물을 찾아주세요.",
        assistant_message=assistant_message,
        filters=session_context.last_filters if session_context else SessionFilters(),
        existing_summary=session_context.summary if session_context else "",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=resolved_session_id,
        assistant_message=assistant_message,
        agent_reasoning=agent_reasoning,
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
    del text_weight
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

    if use_agent and (text_vec is not None or image_vec is not None):
        agent_query = query if query else "이미지로 유사한 분실물을 찾아주세요."
        try:
            agent_result = await run_search_agent(
                user_query=agent_query,
                embedding_service=embedding,
                vector_store=vector_store,
                top_k=top_k,
                session_id=resolved_session_id,
                session_context=session_context,
                image_vec=image_vec,
            )
            results = agent_result.items
            agent_reasoning = agent_result.reasoning
            query_metadata = agent_result.metadata
        except Exception as exc:
            logger.warning("에이전트 실행 실패, 원시 검색으로 폴백: %s", exc)
            if text_vec is not None and image_vec is not None:
                results = await vector_store.search_combined(text_vec=text_vec, image_vec=image_vec, top_k=top_k)
            elif image_vec is not None:
                results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
            else:
                assert text_vec is not None
                results = await vector_store.search_by_text(text_vec, top_k=top_k)
    elif text_vec is not None and image_vec is not None:
        results = await vector_store.search_combined(text_vec=text_vec, image_vec=image_vec, top_k=top_k)
    elif image_vec is not None:
        results = await vector_store.search_by_image(image_vec=image_vec, top_k=top_k)
    else:
        assert text_vec is not None
        results = await vector_store.search_by_text(text_vec, top_k=top_k)

    assistant_message = _build_assistant_message(
        query=query or "복합 검색",
        results=results,
        agent_reasoning=agent_reasoning,
    )
    if query:
        await _persist_session_turn(
            session_service,
            resolved_session_id,
            user_query=query,
            assistant_message=assistant_message,
            filters=session_context.last_filters if session_context else SessionFilters(),
            existing_summary=session_context.summary if session_context else "",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        items=results,
        total=len(results),
        session_id=resolved_session_id,
        assistant_message=assistant_message,
        agent_reasoning=agent_reasoning,
        query_metadata=query_metadata,
        search_time_ms=round(elapsed_ms, 2),
    )
