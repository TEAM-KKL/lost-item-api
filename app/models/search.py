"""검색 요청/응답 Pydantic 모델"""

from pydantic import BaseModel, Field


# ──────────────────────────── 요청 모델 ────────────────────────────

class TextSearchRequest(BaseModel):
    """텍스트로 분실물 검색"""

    query: str = Field(description="분실물 묘사 텍스트 (한국어)", min_length=1)
    top_k: int = Field(default=10, ge=1, le=50, description="반환할 결과 수")
    use_agent: bool = Field(
        default=True,
        description="True: PydanticAI 에이전트 경유 (쿼리 분석 + 추론), False: 원시 벡터 검색",
    )
    filter_category: str | None = Field(default=None, description="물품분류명 필터 (예: 지갑 > 기타 지갑)")
    filter_date_from: str | None = Field(default=None, description="습득일 시작 (YYYY-MM-DD)")
    filter_date_to: str | None = Field(default=None, description="습득일 종료 (YYYY-MM-DD)")


class CombinedSearchRequest(BaseModel):
    """텍스트 + 이미지 복합 검색"""

    query: str | None = Field(default=None, description="분실물 묘사 텍스트")
    top_k: int = Field(default=10, ge=1, le=50)
    text_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="텍스트 벡터 가중치 (이미지 가중치 = 1 - text_weight)",
    )
    use_agent: bool = Field(default=True)


# ──────────────────────────── 응답 모델 ────────────────────────────

class LostItemResult(BaseModel):
    """검색 결과 단일 항목"""

    atc_id: str = Field(description="관리ID")
    fd_prdt_nm: str = Field(description="물품명")
    fd_sbjt: str = Field(description="게시제목")
    prdt_cl_nm: str = Field(description="물품분류명")
    dep_place: str = Field(description="보관장소(경찰서)")
    fd_ymd: str = Field(description="습득일자")
    image_url: str | None = Field(default=None, description="습득물 사진 URL")
    score: float = Field(description="유사도 점수 (0~1, 높을수록 유사)")
    matched_via: str = Field(description="매칭 경로: text_vec | image_vec | combined")


class SearchMetadata(BaseModel):
    """AI 에이전트가 추출한 검색 메타데이터"""

    item_type: str | None = Field(default=None, description="물품 종류 (지갑, 핸드폰, 가방 등)")
    color: str | None = Field(default=None, description="색상")
    material: str | None = Field(default=None, description="재질")
    brand: str | None = Field(default=None, description="브랜드")
    location_hint: str | None = Field(default=None, description="잃어버린 장소 힌트")
    date_hint: str | None = Field(default=None, description="잃어버린 날짜 힌트")


class SearchResponse(BaseModel):
    """검색 API 응답"""

    items: list[LostItemResult] = Field(description="검색 결과 목록")
    total: int = Field(description="반환된 결과 수")
    agent_reasoning: str | None = Field(default=None, description="AI 에이전트 검색 전략 설명 (한국어)")
    query_metadata: SearchMetadata | None = Field(default=None, description="에이전트가 추출한 메타데이터")
    search_time_ms: float = Field(description="검색 소요 시간 (밀리초)")
