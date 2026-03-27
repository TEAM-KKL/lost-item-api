"""PydanticAI 에이전트 - vector_search 툴을 직접 호출하는 방식"""

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent.prompts import LOST_ITEM_SEARCH_PROMPT
from app.models.search import LostItemResult, SearchMetadata
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


# ──────────────────────────── 의존성 (에이전트에 주입) ────────────────────────────

@dataclass
class SearchDeps:
    """에이전트 실행 중 공유되는 의존성 및 결과 누산기"""

    embedding_service: EmbeddingService
    vector_store: VectorStoreService
    top_k: int
    filter_category: str | None = None
    filter_date_from: str | None = None
    filter_date_to: str | None = None
    # 툴 호출마다 누산 (atc_id 기준 중복 제거, 높은 점수 유지)
    collected_items: dict[str, LostItemResult] = field(default_factory=dict)


# ──────────────────────────── 에이전트 결과 타입 ────────────────────────────

class AgentResult(BaseModel):
    """run_search_agent가 반환하는 최종 결과"""

    refined_query: str
    metadata: SearchMetadata
    items: list[LostItemResult]
    reasoning: str


# ──────────────────────────── 에이전트 팩토리 (지연 초기화) ────────────────────────────

_agent: Agent[SearchDeps, str] | None = None


def _get_agent() -> Agent[SearchDeps, str]:
    """
    에이전트 싱글톤 반환.
    첫 호출 시 .env가 이미 로드된 이후이므로 API 키를 안전하게 읽을 수 있음.
    """
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _build_agent() -> Agent[SearchDeps, str]:
    from app.config import get_settings
    settings = get_settings()

    model = OpenAIModel(
        settings.openai_model,
        provider=OpenAIProvider(api_key=settings.openai_api_key),
    )

    agent: Agent[SearchDeps, str] = Agent(
        model=model,
        deps_type=SearchDeps,
        output_type=str,
        system_prompt=LOST_ITEM_SEARCH_PROMPT,
        retries=1,
    )

    # ── 툴 등록 ──────────────────────────────────────────────────────────────

    @agent.tool
    async def vector_search(
        ctx: RunContext[SearchDeps],
        query: str,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """
        한국 경찰청 습득물 데이터베이스에서 의미론적 유사도 검색을 수행합니다.
        결과가 부족하거나 유사도가 낮으면 다른 키워드로 여러 번 호출할 수 있습니다.

        Args:
            query: 검색 키워드 (한국어, 5단어 이내). 물품명+색상+재질 조합. 장소·날짜 제외.
                   예: '검정 가죽 지갑', '흰색 에어팟 케이스', '삼성 갤럭시 S24'
            category: 물품 분류 필터 (선택). 예: '지갑', '핸드폰', '가방'
            date_from: 습득일 시작 필터 (YYYY-MM-DD, 선택)
            date_to: 습득일 종료 필터 (YYYY-MM-DD, 선택)
        """
        deps = ctx.deps

        effective_category = deps.filter_category or category
        effective_date_from = deps.filter_date_from or date_from
        effective_date_to = deps.filter_date_to or date_to

        query_vec = await deps.embedding_service.encode_text(query)
        results = await deps.vector_store.search_by_text(
            query_vec=query_vec,
            top_k=deps.top_k * 2,
            filter_category=effective_category,
            filter_date_from=effective_date_from,
            filter_date_to=effective_date_to,
        )

        logger.info(
            "vector_search 호출: query=%r, category=%s → %d건",
            query, effective_category, len(results),
        )

        for item in results:
            if item.atc_id not in deps.collected_items or item.score > deps.collected_items[item.atc_id].score:
                deps.collected_items[item.atc_id] = item

        if not results:
            return "검색 결과 없음"

        lines = [f"검색 결과 총 {len(results)}건:"]
        for i, r in enumerate(results[:5]):
            lines.append(
                f"[{i+1}] {r.fd_prdt_nm} | {r.prdt_cl_nm} | {r.dep_place} | {r.fd_ymd} | 유사도 {r.score:.3f}"
            )
        if len(results) > 5:
            lines.append(f"... 외 {len(results) - 5}건")

        return "\n".join(lines)

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
    deps = SearchDeps(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=top_k,
        filter_category=filter_category,
        filter_date_from=filter_date_from,
        filter_date_to=filter_date_to,
    )

    result = await _get_agent().run(user_query, deps=deps)
    reasoning: str = result.output

    items = sorted(deps.collected_items.values(), key=lambda x: x.score, reverse=True)[:top_k]

    last_query = user_query
    last_category: str | None = filter_category
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if getattr(part, "tool_name", None) == "vector_search":
                args = getattr(part, "args", {})
                if isinstance(args, dict):
                    last_query = args.get("query", last_query)
                    last_category = filter_category or args.get("category")

    return AgentResult(
        refined_query=last_query,
        metadata=SearchMetadata(item_type=last_category),
        items=items,
        reasoning=reasoning,
    )
