"""데이터 수집(인제스트) 작업 모델"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """인제스트 요청"""

    start_ymd: str = Field(
        description="수집 시작일 (YYYY-MM-DD 또는 YYYYMMDD)",
        examples=["2024-01-01"],
    )
    end_ymd: str = Field(
        description="수집 종료일 (YYYY-MM-DD 또는 YYYYMMDD)",
        examples=["2024-12-31"],
    )
    num_of_rows: int = Field(
        default=100,
        ge=1,
        le=100,
        description="페이지당 항목 수 (최대 100)",
    )
    force_reimport: bool = Field(
        default=False,
        description="True이면 이미 존재하는 항목도 재임베딩/재저장",
    )


class IngestJobState(BaseModel):
    """인제스트 작업 상태 (메모리 내 추적)"""

    job_id: str
    status: IngestStatus = IngestStatus.PENDING
    request: IngestRequest
    pages_fetched: int = 0
    total_pages: int = 0
    items_processed: int = 0
    items_upserted: int = 0
    items_with_images: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def progress_pct(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return round(self.pages_fetched / self.total_pages * 100, 1)


class IngestResponse(BaseModel):
    """인제스트 시작 응답 (202 Accepted)"""

    job_id: str
    status: IngestStatus
    message: str


class IngestStatusResponse(BaseModel):
    """인제스트 작업 상태 조회 응답"""

    job_id: str
    status: IngestStatus
    progress_pct: float
    pages_fetched: int
    total_pages: int
    items_processed: int
    items_upserted: int
    items_with_images: int
    errors: list[str]
    started_at: datetime | None
    completed_at: datetime | None
