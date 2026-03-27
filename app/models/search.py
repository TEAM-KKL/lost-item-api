"""Search request and response models."""

from pydantic import BaseModel, Field


class TextSearchRequest(BaseModel):
    """Text-based lost item search request."""

    query: str = Field(description="분실물 설명 텍스트", min_length=1)
    top_k: int = Field(default=10, ge=1, le=50, description="반환할 결과 수")
    use_agent: bool = Field(
        default=True,
        description="True면 GPT 에이전트가 vector_search 툴로 반복 검색하고, False면 원시 벡터 검색을 수행합니다.",
    )
    session_id: str | None = Field(default=None, description="검색 대화 세션 ID")
    filter_category: str | None = Field(default=None, description="물품분류명 필터")
    filter_date_from: str | None = Field(default=None, description="습득일 시작 (YYYY-MM-DD)")
    filter_date_to: str | None = Field(default=None, description="습득일 종료 (YYYY-MM-DD)")


class CombinedSearchRequest(BaseModel):
    """Combined text and image search request."""

    query: str | None = Field(default=None, description="분실물 설명 텍스트")
    top_k: int = Field(default=10, ge=1, le=50)
    text_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="텍스트 벡터 가중치 (이미지 가중치 = 1 - text_weight)",
    )
    use_agent: bool = Field(default=True)
    session_id: str | None = Field(default=None, description="검색 대화 세션 ID")


class LostItemResult(BaseModel):
    """Single search result item."""

    atc_id: str = Field(description="관리 ID")
    fd_prdt_nm: str = Field(description="물품명")
    fd_sbjt: str = Field(description="게시 제목")
    prdt_cl_nm: str = Field(description="물품분류명")
    dep_place: str = Field(description="보관 장소")
    fd_ymd: str = Field(description="습득일자")
    image_url: str | None = Field(default=None, description="습득물 사진 URL")
    score: float = Field(description="유사도 점수")
    matched_via: str = Field(description="매칭 경로")


class SearchMetadata(BaseModel):
    """Metadata inferred by the AI search agent."""

    item_type: str | None = Field(default=None, description="물품 종류")
    color: str | None = Field(default=None, description="색상")
    material: str | None = Field(default=None, description="재질")
    brand: str | None = Field(default=None, description="브랜드")
    location_hint: str | None = Field(default=None, description="분실 장소 힌트")
    date_hint: str | None = Field(default=None, description="분실 날짜 힌트")


class SearchResponse(BaseModel):
    """Search API response."""

    items: list[LostItemResult] = Field(description="검색 결과 목록")
    total: int = Field(description="반환된 결과 수")
    session_id: str | None = Field(default=None, description="검색 대화 세션 ID")
    agent_reasoning: str | None = Field(default=None, description="AI 에이전트 검색 전략 요약")
    query_metadata: SearchMetadata | None = Field(default=None, description="AI가 추출한 쿼리 메타데이터")
    search_time_ms: float = Field(description="검색 소요 시간 (밀리초)")
