"""경찰청 공공데이터 포털 습득물 API 응답 모델"""

from pydantic import BaseModel, Field, field_validator


class PoliceAPIItem(BaseModel):
    """습득물 단일 항목"""

    atcId: str = Field(description="관리ID")
    depPlace: str = Field(description="보관장소")
    fdFilePathImg: str | None = Field(default=None, description="습득물 사진 이미지 URL")
    fdPrdtNm: str = Field(description="물품명")
    fdSbjt: str = Field(description="게시제목")
    fdSn: str = Field(description="습득순번")
    fdYmd: str = Field(description="습득일자 (YYYY-MM-DD)")
    prdtClNm: str = Field(description="물품분류명 (예: 지갑 > 기타 지갑)")
    rnum: str = Field(description="일련번호")

    @field_validator("fdFilePathImg", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: str | None) -> str | None:
        """빈 문자열을 None으로 변환"""
        if v == "" or v is None:
            return None
        return v

    def build_text_for_embedding(self) -> str:
        """CLIP text_vec 생성에 사용할 텍스트 조합"""
        parts = [self.fdPrdtNm, self.fdSbjt, self.prdtClNm]
        return " ".join(p for p in parts if p and p.strip())


class PoliceAPIBody(BaseModel):
    """API 응답 body"""

    totalCount: int = Field(default=0)
    numOfRows: int = Field(default=10)
    pageNo: int = Field(default=1)
    items: list[PoliceAPIItem] = Field(default_factory=list)


class PoliceAPIResponse(BaseModel):
    """최상위 API 응답"""

    resultCode: str = Field(description="결과코드 (00 = 정상)")
    resultMag: str = Field(description="결과메시지")
    body: PoliceAPIBody = Field(default_factory=PoliceAPIBody)

    @property
    def is_success(self) -> bool:
        return self.resultCode == "00"

    @property
    def total_count(self) -> int:
        return self.body.totalCount

    @property
    def items(self) -> list[PoliceAPIItem]:
        return self.body.items
