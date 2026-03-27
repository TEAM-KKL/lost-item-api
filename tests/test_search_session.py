from datetime import datetime, timezone

from app.agent.agent import SearchDeps, build_agent_input
from app.api.routes.search import _build_assistant_message, _filters_to_dict, _merge_filters
from app.models.search import LostItemResult, SessionHistoryResponse, SessionMessageResponse
from app.services.search_session import (
    SessionContext,
    SessionFilters,
    SessionMessage,
    summarize_messages,
)


def test_build_agent_input_reads_session_context_from_deps() -> None:
    deps = SearchDeps(
        embedding_service=None,  # type: ignore[arg-type]
        vector_store=None,  # type: ignore[arg-type]
        top_k=10,
        session_id="session-1",
        session_context=SessionContext(
            session_id="session-1",
            summary="Previous search was for a black wallet near Gangnam station.",
            recent_messages=[
                SessionMessage(role="user", content="I lost it last week."),
                SessionMessage(role="assistant", content="I will search for black leather wallets first."),
            ],
            last_filters=SessionFilters(
                filter_category="wallet",
                filter_date_from="2026-03-20",
                filter_date_to="2026-03-27",
            ),
        ),
    )

    prompt = build_agent_input("Find that wallet again", deps)

    assert "[session_summary]" in prompt
    assert "[recent_messages]" in prompt
    assert "[inherited_filters]" in prompt
    assert "[current_user_request]" in prompt
    assert "Find that wallet again" in prompt
    assert "wallet" in prompt


def test_merge_filters_prefers_explicit_request_values() -> None:
    merged = _merge_filters(
        SessionFilters(
            filter_category="bag",
            filter_date_from=None,
            filter_date_to="2026-03-27",
        ),
        SessionFilters(
            filter_category="wallet",
            filter_date_from="2026-03-20",
            filter_date_to="2026-03-26",
        ),
    )

    assert merged.filter_category == "bag"
    assert merged.filter_date_from == "2026-03-20"
    assert merged.filter_date_to == "2026-03-27"


def test_summarize_messages_keeps_compact_history() -> None:
    summary = summarize_messages(
        "Earlier summary",
        [
            SessionMessage(role="user", content="I am looking for a black wallet lost near Gangnam station."),
            SessionMessage(role="assistant", content="I will search wallet and card holder results first."),
        ],
    )

    assert "Earlier summary" in summary
    assert "user:" in summary
    assert "assistant:" in summary


def test_filters_to_dict_keeps_expected_shape() -> None:
    filters = _filters_to_dict(
        SessionFilters(
            filter_category="wallet",
            filter_date_from="2026-03-20",
            filter_date_to=None,
        )
    )

    assert filters == {
        "filter_category": "wallet",
        "filter_date_from": "2026-03-20",
        "filter_date_to": None,
    }


def test_session_history_response_serializes_messages() -> None:
    message = SessionMessageResponse(
        role="user",
        content="Find my wallet",
        created_at=datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc).isoformat(),
    )
    response = SessionHistoryResponse(
        session_id="session-1",
        summary="Earlier summary",
        last_filters={"filter_category": "wallet"},
        messages=[message],
    )

    assert response.session_id == "session-1"
    assert response.messages[0].role == "user"
    assert response.last_filters["filter_category"] == "wallet"


def test_build_assistant_message_prefers_agent_reasoning() -> None:
    message = _build_assistant_message(
        query="검정 자켓",
        results=[],
        agent_reasoning="부산역 보관소에 검정 자켓 후보가 1건 있습니다. 확인해 보세요.",
    )

    assert message == "부산역 보관소에 검정 자켓 후보가 1건 있습니다. 확인해 보세요."


def test_build_assistant_message_builds_plain_response_without_agent_reasoning() -> None:
    message = _build_assistant_message(
        query="검정 자켓",
        results=[
            LostItemResult(
                atc_id="1",
                fd_prdt_nm="검정 자켓",
                fd_sbjt="검정 자켓 습득",
                prdt_cl_nm="의류",
                dep_place="부산역",
                fd_ymd="2026-03-23",
                image_url=None,
                score=0.783,
                matched_via="text_vec",
            )
        ],
    )

    assert "'검정 자켓'와 관련된 결과 1건을 찾았습니다." in message
    assert "1. 검정 자켓 / 부산역 / 2026-03-23 / 유사도 0.783" in message
