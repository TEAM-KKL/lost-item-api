"""Qdrant 벡터 DB 연동 서비스"""

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from app.models.mongo_item import MongoLostItem
from app.models.police_api import PoliceAPIItem
from app.models.search import LostItemResult

logger = logging.getLogger(__name__)

# UUID 네임스페이스 (deterministic point ID 생성용)
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# RRF 상수 (Reciprocal Rank Fusion) — search_combined에서만 사용
_RRF_K = 60

# 배치 upsert 크기
_BATCH_SIZE = 100


def make_point_id(atc_id: str) -> str:
    """atcId → 결정론적 UUID (중복 방지, 멱등 upsert 보장)"""
    return str(uuid.uuid5(_UUID_NAMESPACE, atc_id))


@dataclass
class SearchHit:
    point_id: str
    score: float
    payload: dict
    matched_via: str


class VectorStoreService:
    """
    Qdrant 컬렉션 `lost_items` 관리.

    Named vectors:
    - text_vec  (768-dim COSINE): multilingual-mpnet 텍스트 임베딩 (항상 존재)
    - image_vec (512-dim COSINE): CLIP 이미지 임베딩 (사진 있을 때만)

    텍스트 검색은 text_vec만, 이미지 검색은 image_vec만 사용.
    복합 검색(search_combined)은 두 벡터를 각각 검색 후 RRF로 병합.
    """

    COLLECTION = "lost_items"

    def __init__(self, client: AsyncQdrantClient):
        self._client = client

    # ──────────────────────────── 컬렉션 초기화 ────────────────────────────

    async def ensure_collection(self) -> None:
        """컬렉션이 없으면 생성 (멱등)"""
        exists = await self._client.collection_exists(self.COLLECTION)
        if exists:
            logger.info("Qdrant 컬렉션 '%s' 이미 존재", self.COLLECTION)
            return

        logger.info("Qdrant 컬렉션 '%s' 생성 중...", self.COLLECTION)
        await self._client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config={
                "text_vec": VectorParams(size=768, distance=Distance.COSINE),   # multilingual-mpnet
                "image_vec": VectorParams(size=512, distance=Distance.COSINE),  # CLIP
            },
        )

        # 페이로드 인덱스 (필터 검색 성능 향상)
        for field, schema in [
            ("fd_ymd", PayloadSchemaType.KEYWORD),
            ("prdt_cl_nm", PayloadSchemaType.KEYWORD),
            ("dep_place", PayloadSchemaType.KEYWORD),
            ("has_image", PayloadSchemaType.BOOL),
        ]:
            await self._client.create_payload_index(
                collection_name=self.COLLECTION,
                field_name=field,
                field_schema=schema,
            )

        logger.info("Qdrant 컬렉션 '%s' 생성 완료", self.COLLECTION)

    # ──────────────────────────── Upsert ────────────────────────────

    async def upsert_item(
        self,
        item: PoliceAPIItem,
        text_vec: list[float],
        image_vec: list[float] | None,
    ) -> None:
        """단일 항목 upsert (배치 사용 권장)"""
        await self.upsert_batch([(item, text_vec, image_vec)])

    async def upsert_batch(
        self,
        items: list[tuple[PoliceAPIItem, list[float], list[float] | None]],
    ) -> int:
        """
        배치 upsert. (item, text_vec, image_vec | None) 튜플 리스트를 받아
        Qdrant에 저장하고 upsert된 건수를 반환합니다.
        """
        points = []
        for item, text_vec, image_vec in items:
            vectors: dict[str, list[float]] = {"text_vec": text_vec}
            if image_vec is not None:
                vectors["image_vec"] = image_vec

            points.append(
                PointStruct(
                    id=make_point_id(item.atcId),
                    vector=vectors,
                    payload={
                        "atc_id": item.atcId,
                        "fd_prdt_nm": item.fdPrdtNm,
                        "fd_sbjt": item.fdSbjt,
                        "prdt_cl_nm": item.prdtClNm,
                        "dep_place": item.depPlace,
                        "fd_ymd": item.fdYmd,
                        "fd_file_path_img": item.fdFilePathImg,
                        "has_image": image_vec is not None,
                    },
                )
            )

        if points:
            await self._client.upsert(
                collection_name=self.COLLECTION,
                points=points,
                wait=True,
            )
        return len(points)

    async def upsert_mongo_batch(
        self,
        items: list[tuple["MongoLostItem", list[float], list[float] | None]],
    ) -> int:
        """
        MongoDB 문서 배치 upsert.
        페이로드 필드를 경찰청 API와 동일한 키로 정규화하여 저장합니다.
        """
        points = []
        for item, text_vec, image_vec in items:
            vectors: dict[str, list[float]] = {"text_vec": text_vec}
            if image_vec is not None:
                vectors["image_vec"] = image_vec

            points.append(
                PointStruct(
                    id=make_point_id(item.mng_id),
                    vector=vectors,
                    payload={
                        "atc_id": item.mng_id,
                        "fd_prdt_nm": item.item_cn,
                        "fd_sbjt": item.pstg_ttl,
                        "prdt_cl_nm": item.category,
                        "dep_place": item.kpng_plc_nm,
                        "fd_ymd": item.lost_cmdty_pkup_ymd,
                        "fd_file_path_img": item.image_path or None,
                        "has_image": image_vec is not None,
                        "sgg_nm": item.sgg_nm,
                        "source": "mongodb",
                    },
                )
            )

        if points:
            await self._client.upsert(
                collection_name=self.COLLECTION,
                points=points,
                wait=True,
            )
        return len(points)

    # ──────────────────────────── 검색 ────────────────────────────

    def _build_filter(
        self,
        filter_category: str | None = None,
        filter_date_from: str | None = None,
        filter_date_to: str | None = None,
    ) -> Filter | None:
        conditions = []
        if filter_category:
            conditions.append(
                FieldCondition(key="prdt_cl_nm", match=MatchValue(value=filter_category))
            )
        if filter_date_from or filter_date_to:
            conditions.append(
                FieldCondition(
                    key="fd_ymd",
                    range=Range(
                        gte=filter_date_from,
                        lte=filter_date_to,
                    ),
                )
            )
        return Filter(must=conditions) if conditions else None

    def _points_to_results(self, points: list, via: str) -> list[LostItemResult]:
        """Qdrant 검색 결과를 LostItemResult 리스트로 변환"""
        results = []
        for hit in points:
            p = hit.payload or {}
            results.append(
                LostItemResult(
                    atc_id=p.get("atc_id", ""),
                    fd_prdt_nm=p.get("fd_prdt_nm", ""),
                    fd_sbjt=p.get("fd_sbjt", ""),
                    prdt_cl_nm=p.get("prdt_cl_nm", ""),
                    dep_place=p.get("dep_place", ""),
                    fd_ymd=p.get("fd_ymd", ""),
                    image_url=p.get("fd_file_path_img"),
                    score=round(hit.score, 6),
                    matched_via=via,
                )
            )
        return results

    async def search_by_text(
        self,
        query_vec: list[float],
        top_k: int = 10,
        filter_category: str | None = None,
        filter_date_from: str | None = None,
        filter_date_to: str | None = None,
    ) -> list[LostItemResult]:
        """텍스트 벡터(768-dim)로 text_vec 검색"""
        qdrant_filter = self._build_filter(filter_category, filter_date_from, filter_date_to)

        result = await self._client.query_points(
            collection_name=self.COLLECTION,
            query=query_vec,
            using="text_vec",
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        return self._points_to_results(result.points, "text_vec")

    async def search_by_image(
        self,
        image_vec: list[float],
        top_k: int = 10,
    ) -> list[LostItemResult]:
        """이미지 벡터(512-dim)로 image_vec 검색"""
        result = await self._client.query_points(
            collection_name=self.COLLECTION,
            query=image_vec,
            using="image_vec",
            limit=top_k,
            with_payload=True,
        )
        return self._points_to_results(result.points, "image_vec")

    async def search_combined(
        self,
        text_vec: list[float],
        image_vec: list[float],
        top_k: int = 10,
        filter_category: str | None = None,
    ) -> list[LostItemResult]:
        """텍스트→text_vec + 이미지→image_vec 각각 검색 후 RRF 병합"""
        qdrant_filter = self._build_filter(filter_category)

        text_result = await self._client.query_points(
            collection_name=self.COLLECTION,
            query=text_vec,
            using="text_vec",
            limit=top_k * 2,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        image_result = await self._client.query_points(
            collection_name=self.COLLECTION,
            query=image_vec,
            using="image_vec",
            limit=top_k * 2,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        return self._rrf_merge(text_result.points, "text_vec", image_result.points, "image_vec", top_k)

    # ──────────────────────────── RRF 병합 ────────────────────────────

    def _rrf_merge(
        self,
        primary_hits: list,
        primary_label: str,
        secondary_hits: list,
        secondary_label: str,
        top_k: int,
    ) -> list[LostItemResult]:
        """
        Reciprocal Rank Fusion (RRF) 방식으로 두 결과 목록 병합.
        score = sum(1 / (k + rank)) where k=60
        """
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        matched_via: dict[str, str] = {}

        for rank, hit in enumerate(primary_hits):
            pid = str(hit.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            payloads[pid] = hit.payload or {}
            matched_via[pid] = primary_label

        for rank, hit in enumerate(secondary_hits):
            pid = str(hit.id)
            prev = rrf_scores.get(pid, 0.0)
            rrf_scores[pid] = prev + 1.0 / (_RRF_K + rank + 1)
            if pid not in payloads:
                payloads[pid] = hit.payload or {}
            if pid in matched_via and matched_via[pid] != secondary_label:
                matched_via[pid] = "combined"
            elif pid not in matched_via:
                matched_via[pid] = secondary_label

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for pid, score in ranked:
            p = payloads.get(pid, {})
            results.append(
                LostItemResult(
                    atc_id=p.get("atc_id", ""),
                    fd_prdt_nm=p.get("fd_prdt_nm", ""),
                    fd_sbjt=p.get("fd_sbjt", ""),
                    prdt_cl_nm=p.get("prdt_cl_nm", ""),
                    dep_place=p.get("dep_place", ""),
                    fd_ymd=p.get("fd_ymd", ""),
                    image_url=p.get("fd_file_path_img"),
                    score=round(score, 6),
                    matched_via=matched_via.get(pid, "unknown"),
                )
            )
        return results

    # ──────────────────────────── 최근 목록 ────────────────────────────

    async def get_recent_items(
        self,
        limit: int = 20,
        offset: int = 0,
        filter_category: str | None = None,
        filter_date_from: str | None = None,
        filter_date_to: str | None = None,
    ) -> tuple[list[LostItemResult], bool]:
        """
        최근 등록된 분실물 목록 반환 (fd_ymd 내림차순).

        날짜 필터를 지정하지 않으면 최근 30일 데이터만 조회합니다.
        최대 1000건을 fetch한 뒤 Python에서 정렬 후 페이지네이션을 적용합니다.
        """
        from datetime import date, timedelta

        if not filter_date_from and not filter_date_to:
            filter_date_from = (date.today() - timedelta(days=30)).isoformat()

        # fd_ymd는 KEYWORD 인덱스라 Qdrant Range 필터 불가 → 카테고리만 Qdrant에 위임
        qdrant_filter = self._build_filter(filter_category)

        _MAX_FETCH = 1000
        all_points: list = []
        current_offset = None

        while len(all_points) < _MAX_FETCH:
            batch, current_offset = await self._client.scroll(
                collection_name=self.COLLECTION,
                scroll_filter=qdrant_filter,
                limit=min(256, _MAX_FETCH - len(all_points)),
                offset=current_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if current_offset is None:
                break

        # 날짜 필터링 및 정렬은 Python에서 처리
        def _in_range(point: object) -> bool:
            fd_ymd = (getattr(point, "payload", None) or {}).get("fd_ymd", "")
            if filter_date_from and fd_ymd < filter_date_from:
                return False
            if filter_date_to and fd_ymd > filter_date_to:
                return False
            return True

        all_points = [p for p in all_points if _in_range(p)]
        all_points.sort(
            key=lambda p: (p.payload or {}).get("fd_ymd", ""),
            reverse=True,
        )

        page = all_points[offset: offset + limit]
        has_next = len(all_points) > offset + limit

        items = []
        for hit in page:
            p = hit.payload or {}
            items.append(
                LostItemResult(
                    atc_id=p.get("atc_id", ""),
                    fd_prdt_nm=p.get("fd_prdt_nm", ""),
                    fd_sbjt=p.get("fd_sbjt", ""),
                    prdt_cl_nm=p.get("prdt_cl_nm", ""),
                    dep_place=p.get("dep_place", ""),
                    fd_ymd=p.get("fd_ymd", ""),
                    image_url=p.get("fd_file_path_img"),
                    score=1.0,
                    matched_via="recent",
                )
            )

        return items, has_next

    # ──────────────────────────── 통계 ────────────────────────────

    async def get_collection_info(self) -> dict:
        """컬렉션 통계 정보 반환"""
        info = await self._client.get_collection(self.COLLECTION)
        return {
            "vectors_count": getattr(info, "vectors_count", None),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "points_count": getattr(info, "points_count", None),
            "status": str(getattr(info, "status", "unknown")),
        }
