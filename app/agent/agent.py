"""PydanticAI search agent with optional session context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent.prompts import LOST_ITEM_SEARCH_PROMPT
from app.models.search import LostItemResult, SearchMetadata
from app.services.embedding import EmbeddingService
from app.services.search_session import SessionFilters, SessionMessage
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class SearchDeps:
    """Shared dependencies and search accumulator for one agent run."""

    embedding_service: EmbeddingService
    vector_store: VectorStoreService
    top_k: int
    filter_category: str | None = None
    filter_date_from: str | None = None
    filter_date_to: str | None = None
    collected_items: dict[str, LostItemResult] = field(default_factory=dict)


class AgentResult(BaseModel):
    """Final agent output returned to the API layer."""

    refined_query: str
    metadata: SearchMetadata
    items: list[LostItemResult]
    reasoning: str


def build_agent_input(
    user_query: str,
    recent_messages: list[SessionMessage] | None = None,
    summary: str = "",
    inherited_filters: SessionFilters | None = None,
) -> str:
    """Compose the current user request with session context for the agent."""
    sections: list[str] = []

    if summary.strip():
        sections.append(f"[세션 요약]\n{summary.strip()}")

    if recent_messages:
        history_lines = []
        for message in recent_messages:
            role_label = "사용자" if message.role == "user" else "어시스턴트"
            history_lines.append(f"- {role_label}: {message.content}")
        sections.append("[최근 대화]\n" + "\n".join(history_lines))

    if inherited_filters and any(
        [
            inherited_filters.filter_category,
            inherited_filters.filter_date_from,
            inherited_filters.filter_date_to,
        ]
    ):
        filter_lines = []
        if inherited_filters.filter_category:
            filter_lines.append(f"- category: {inherited_filters.filter_category}")
        if inherited_filters.filter_date_from:
            filter_lines.append(f"- date_from: {inherited_filters.filter_date_from}")
        if inherited_filters.filter_date_to:
            filter_lines.append(f"- date_to: {inherited_filters.filter_date_to}")
        sections.append("[유지 중인 검색 필터]\n" + "\n".join(filter_lines))

    sections.append(f"[현재 사용자 요청]\n{user_query}")
    return "\n\n".join(sections)


_agent: Agent[SearchDeps, str] | None = None


def _get_agent() -> Agent[SearchDeps, str]:
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

    @agent.tool
    async def vector_search(
        ctx: RunContext[SearchDeps],
        query: str,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
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
            "vector_search 호출: query=%r, category=%s -> %d건",
            query,
            effective_category,
            len(results),
        )

        for item in results:
            if item.atc_id not in deps.collected_items or item.score > deps.collected_items[item.atc_id].score:
                deps.collected_items[item.atc_id] = item

        if not results:
            return "검색 결과 없음"

        lines = [f"검색 결과 총 {len(results)}건:"]
        for index, result in enumerate(results[:5], start=1):
            lines.append(
                f"[{index}] {result.fd_prdt_nm} | {result.prdt_cl_nm} | {result.dep_place} | {result.fd_ymd} | 유사도 {result.score:.3f}"
            )
        if len(results) > 5:
            lines.append(f"... 외 {len(results) - 5}건")
        return "\n".join(lines)

    return agent


async def run_search_agent(
    user_query: str,
    embedding_service: EmbeddingService,
    vector_store: VectorStoreService,
    top_k: int = 10,
    filter_category: str | None = None,
    filter_date_from: str | None = None,
    filter_date_to: str | None = None,
    session_summary: str = "",
    recent_messages: list[SessionMessage] | None = None,
    inherited_filters: SessionFilters | None = None,
) -> AgentResult:
    deps = SearchDeps(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=top_k,
        filter_category=filter_category,
        filter_date_from=filter_date_from,
        filter_date_to=filter_date_to,
    )

    agent_input = build_agent_input(
        user_query=user_query,
        recent_messages=recent_messages,
        summary=session_summary,
        inherited_filters=inherited_filters,
    )
    result = await _get_agent().run(agent_input, deps=deps)
    reasoning = result.output

    items = sorted(deps.collected_items.values(), key=lambda item: item.score, reverse=True)[:top_k]

    last_query = user_query
    last_category: str | None = filter_category
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
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
