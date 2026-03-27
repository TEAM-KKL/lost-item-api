from app.agent.agent import SearchDeps, build_agent_input
from app.api.routes.search import _merge_filters
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
