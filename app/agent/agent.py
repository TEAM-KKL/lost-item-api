"""PydanticAI 에이전트 - 한국어 자연어로 분실물 검색"""

import logging

from pydantic import BaseModel
from pydantic_ai import Agent

from app.agent.prompts import LOST_ITEM_SEARCH_PROMPT
from app.models.search import LostItemResult, SearchMetadata
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


# ──────────────────────────── 에이전트 출력 타입 ────────────────────────────

class QueryAnalysis(BaseModel):
    """GPT-4o가 반환하는 쿼리 분석 결과 (툴 호출 없음)"""

    refined_query: str
    """검색에 최적화된 한국어 쿼리 (5단어 이내)"""

    metadata: SearchMetadata
    """분실물 메타데이터 (물품종류, 색상, 재질, 브랜드, 장소, 날짜)"""

    reasoning: str
    """검색 전략 한 줄 설명 (한국어)"""


# ──────────────────────────── 에이전트 결과 타입 ────────────────────────────

class AgentResult(BaseModel):
    """run_search_agent가 반환하는 최종 결과"""

    refined_query: str
    metadata: SearchMetadata
    items: list[LostItemResult]
    reasoning: str


# ──────────────────────────── 에이전트 팩토리 (지연 초기화) ────────────────────────────

_agent_instance: Agent | None = None


def get_search_agent() -> Agent:
    """
    검색 에이전트 싱글톤 반환.
    OPENAI_API_KEY 환경변수가 설정된 후에 처음 호출 시 초기화됩니다.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = _build_agent()
    return _agent_instance


def _build_agent() -> Agent:
    """에이전트 인스턴스 생성 (툴 없음 — 쿼리 분석 전용)"""
    from app.config import get_settings
    settings = get_settings()
    model_name = f"openai:{settings.openai_model}"

    agent: Agent[None, QueryAnalysis] = Agent(
        model=model_name,
        system_prompt=LOST_ITEM_SEARCH_PROMPT,
        output_type=QueryAnalysis,
        retries=1,
    )

    return agent


# ──────────────────────────── 에이전트 실행 헬퍼 ────────────────────────────

async def run_search_agent(
    user_query: str,
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
    top_k: int = 10,
    filter_category: str | None = None,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
) -> AgentResult:
    """
    1단계: GPT-4o로 쿼리 분석 (툴 호출 없음, 1회 LLM 호출)
    2단계: 정제된 쿼리로 CLIP 벡터 검색 실행
    """
    agent = get_search_agent()

    # 1단계: 쿼리 분석 (GPT-4o 1회 호출)
    analysis_result = await agent.run(user_query)
    analysis: QueryAnalysis = analysis_result.output
    logger.info("쿼리 분석 완료: refined_query=%s", analysis.refined_query)

    # 2단계: 정제된 쿼리로 벡터 검색
    query_vec = await embedding_service.encode_text(analysis.refined_query)
    items = await vector_store.search_by_text(
        query_vec=query_vec,
        top_k=top_k,
        filter_category=filter_category or analysis.metadata.item_type,
        filter_date_from=filter_date_from,
        filter_date_to=filter_date_to,
    )

    return AgentResult(
        refined_query=analysis.refined_query,
        metadata=analysis.metadata,
        items=items,
        reasoning=analysis.reasoning,
    )
