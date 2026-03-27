"""경찰청 공공데이터포털 습득물 API 비동기 클라이언트"""

import asyncio
import logging
import math
from typing import AsyncGenerator

import httpx

from app.models.police_api import PoliceAPIItem, PoliceAPIResponse

logger = logging.getLogger(__name__)

# 동시 요청 제한 (rate limit 방지)
_CONCURRENT_LIMIT = 5


class PoliceAPIError(Exception):
    pass


class PoliceAPIClient:
    """
    경찰청 습득물 API 비동기 클라이언트.

    - fetch_page(): 단일 페이지 조회
    - fetch_all(): 전체 페이지 AsyncGenerator (스트리밍 방식)
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, base_url: str):
        self._client = http_client
        self._api_key = api_key
        self._base_url = base_url

    def _build_params(
        self,
        page_no: int,
        num_of_rows: int,
        start_ymd: str,
        end_ymd: str,
    ) -> dict:
        # 날짜에서 '-' 제거 (YYYYMMDD 형식으로)
        start = start_ymd.replace("-", "")
        end = end_ymd.replace("-", "")
        return {
            "serviceKey": self._api_key,
            "pageNo": str(page_no),
            "numOfRows": str(num_of_rows),
            "START_YMD": start,
            "END_YMD": end,
            "_type": "json",
        }

    async def fetch_page(
        self,
        page_no: int,
        num_of_rows: int,
        start_ymd: str,
        end_ymd: str,
    ) -> PoliceAPIResponse:
        """단일 페이지 조회"""
        params = self._build_params(page_no, num_of_rows, start_ymd, end_ymd)
        try:
            resp = await self._client.get(self._base_url, params=params, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise PoliceAPIError(f"HTTP 오류 (page {page_no}): {e}") from e

        data = resp.json()

        # 실제 API 응답 구조: {"response": {"header": {...}, "body": {...}}}
        try:
            response_body = data["response"]
            header = response_body["header"]
            body = response_body.get("body", {})

            # items가 없거나 빈 경우 처리
            raw_items = body.get("items", {})
            if not raw_items or raw_items == "":
                items_list = []
            else:
                item_data = raw_items.get("item", [])
                # 단일 항목이면 dict, 복수이면 list
                if isinstance(item_data, dict):
                    items_list = [item_data]
                elif isinstance(item_data, list):
                    items_list = item_data
                else:
                    items_list = []

            parsed_items = []
            for item in items_list:
                try:
                    parsed_items.append(PoliceAPIItem(**item))
                except Exception as e:
                    logger.warning("항목 파싱 실패: %s | 데이터: %s", e, item)

            return PoliceAPIResponse(
                resultCode=header.get("resultCode", "99"),
                resultMag=header.get("resultMsg", ""),
                body={
                    "totalCount": body.get("totalCount", 0),
                    "numOfRows": body.get("numOfRows", num_of_rows),
                    "pageNo": body.get("pageNo", page_no),
                    "items": parsed_items,
                },
            )
        except (KeyError, TypeError) as e:
            raise PoliceAPIError(f"응답 파싱 실패: {e} | 응답: {data}") from e

    async def fetch_all(
        self,
        start_ymd: str,
        end_ymd: str,
        num_of_rows: int = 100,
    ) -> AsyncGenerator[PoliceAPIItem, None]:
        """
        전체 데이터를 페이지별로 순차 조회하는 AsyncGenerator.
        Semaphore로 최대 동시 요청 수를 제한합니다.
        """
        # 1페이지 먼저 조회해서 전체 건수 파악
        first_page = await self.fetch_page(1, num_of_rows, start_ymd, end_ymd)
        if not first_page.is_success:
            raise PoliceAPIError(
                f"API 오류: {first_page.resultCode} - {first_page.resultMag}"
            )

        total_count = first_page.total_count
        total_pages = max(1, math.ceil(total_count / num_of_rows))
        logger.info(
            "경찰청 API: 총 %d건, %d페이지 (날짜: %s ~ %s)",
            total_count,
            total_pages,
            start_ymd,
            end_ymd,
        )

        # 1페이지 결과 먼저 yield
        for item in first_page.items:
            yield item

        if total_pages <= 1:
            return

        # 나머지 페이지 병렬 조회
        semaphore = asyncio.Semaphore(_CONCURRENT_LIMIT)

        async def fetch_with_semaphore(page_no: int) -> list[PoliceAPIItem]:
            async with semaphore:
                try:
                    page = await self.fetch_page(page_no, num_of_rows, start_ymd, end_ymd)
                    return page.items
                except PoliceAPIError as e:
                    logger.error("페이지 %d 조회 실패: %s", page_no, e)
                    return []

        # 5페이지씩 배치로 처리 (메모리 효율)
        remaining_pages = list(range(2, total_pages + 1))
        batch_size = _CONCURRENT_LIMIT * 2  # 10페이지씩

        for i in range(0, len(remaining_pages), batch_size):
            batch = remaining_pages[i : i + batch_size]
            tasks = [fetch_with_semaphore(p) for p in batch]
            results = await asyncio.gather(*tasks)
            for items in results:
                for item in items:
                    yield item
