"""MongoDB crawl.lost_items 컬렉션 문서 모델"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class MongoLostItem(BaseModel):
    """
    MongoDB crawl.lost_items 컬렉션의 단일 문서.

    출처: minwon24.police.go.kr 크롤링 데이터
    이미지: E:/projects/lostitem/downloads/{MNG_ID}_{STRG}.jpg
    """

    model_config = {"populate_by_name": True}

    id: str = Field(alias="_id", description="관리ID (PKUP_CMDTY_MNG_ID와 동일)")
    item_cn: str = Field(alias="ITEM_CN", default="", description="물품명")
    pstg_ttl: str = Field(alias="PSTG_TTL", default="", description="게시제목")
    kpng_plc_nm: str = Field(alias="KPNG_PLC_NM", default="", description="보관장소")
    lost_cmdty_pkup_ymd: str = Field(alias="LOST_CMDTY_PKUP_YMD", default="", description="습득일자")
    pkup_cmdty_lclsf_nm: str = Field(alias="PKUP_CMDTY_LCLSF_NM", default="", description="대분류명")
    pkup_cmdty_mclsf_nm: str = Field(alias="PKUP_CMDTY_MCLSF_NM", default="", description="중분류명")
    sgg_nm: str = Field(alias="SGG_NM", default="", description="시군구명")
    strg_file_path: str = Field(alias="STRG_FILE_PATH", default="", description="이미지 파일 경로 코드")
    image_path: str = Field(default="", description="로컬 이미지 절대경로")
    search_text: str = Field(default="", description="검색용 텍스트 (크롤러가 생성)")

    @field_validator("image_path", mode="before")
    @classmethod
    def normalize_image_path(cls, v: str) -> str:
        if not v:
            return ""
        return str(v)

    @property
    def mng_id(self) -> str:
        return self.id

    @property
    def category(self) -> str:
        parts = [p for p in [self.pkup_cmdty_lclsf_nm, self.pkup_cmdty_mclsf_nm] if p]
        return " > ".join(parts) if parts else ""

    def get_local_image_path(self, downloads_dir: str) -> Path | None:
        """
        다운로드 폴더에서 이미지 파일 경로 반환.
        파일명 형식: {MNG_ID}_{STRG_FILE_PATH}.jpg
        """
        if not self.strg_file_path:
            return None
        filename = f"{self.id}_{self.strg_file_path}.jpg"
        path = Path(downloads_dir) / filename
        return path if path.exists() else None

    def build_text_for_embedding(self) -> str:
        """CLIP text_vec 생성에 사용할 텍스트. search_text가 있으면 우선 사용."""
        if self.search_text:
            return self.search_text
        parts = [self.item_cn, self.pstg_ttl, self.category, self.sgg_nm]
        return " ".join(p for p in parts if p and p.strip())
