"""이미지 파일 서빙 엔드포인트"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])


@router.get("/api/v1/images/{filename}")
async def get_image(filename: str, request: Request) -> FileResponse:
    """
    로컬 downloads 디렉토리에서 이미지 파일을 반환합니다.

    - **filename**: 파일명 (확장자 제외). 예: F2026032600004776_C2026032600105751
    """
    settings = get_settings()
    downloads_dir = Path(settings.downloads_dir).expanduser().resolve()
    image_path = downloads_dir / f"{filename}.jpg"

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {filename}")

    return FileResponse(image_path, media_type="image/jpeg")
