from app.agent.agent import build_agent_input
from app.api.routes.search import _merge_filters
from app.services.search_session import SessionFilters, SessionMessage, summarize_messages


def test_build_agent_input_includes_session_context() -> None:
    prompt = build_agent_input(
        user_query="검정 지갑 찾아줘",
        summary="이전에는 강남역 근처에서 지갑을 찾고 있었다.",
        recent_messages=[
            SessionMessage(role="user", content="지난주 강남역에서 잃어버렸어요."),
            SessionMessage(role="assistant", content="검정 가죽 지갑으로 먼저 찾아볼게요."),
        ],
        inherited_filters=SessionFilters(
            filter_category="지갑",
            filter_date_from="2026-03-20",
            filter_date_to="2026-03-27",
        ),
    )

    assert "[세션 요약]" in prompt
    assert "[최근 대화]" in prompt
    assert "[유지 중인 검색 필터]" in prompt
    assert "[현재 사용자 요청]" in prompt
    assert "검정 지갑 찾아줘" in prompt


def test_merge_filters_prefers_explicit_request_values() -> None:
    merged = _merge_filters(
        SessionFilters(
            filter_category="가방",
            filter_date_from=None,
            filter_date_to="2026-03-27",
        ),
        SessionFilters(
            filter_category="지갑",
            filter_date_from="2026-03-20",
            filter_date_to="2026-03-26",
        ),
    )

    assert merged.filter_category == "가방"
    assert merged.filter_date_from == "2026-03-20"
    assert merged.filter_date_to == "2026-03-27"


def test_summarize_messages_keeps_compact_history() -> None:
    summary = summarize_messages(
        "이전 요약",
        [
            SessionMessage(role="user", content="강남역에서 잃어버린 검은 지갑을 찾고 있어요."),
            SessionMessage(role="assistant", content="검정 가죽 지갑과 카드지갑 쪽으로 검색해보겠습니다."),
        ],
    )

    assert "이전 요약" in summary
    assert "사용자:" in summary
    assert "어시스턴트:" in summary
