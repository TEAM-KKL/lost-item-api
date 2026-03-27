"""임베딩 서비스 - 텍스트(multilingual)와 이미지(CLIP)를 별도 모델로 임베딩"""

import asyncio
import io
import logging

import httpx
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    텍스트와 이미지를 각각 최적화된 모델로 임베딩합니다.

    - 텍스트: paraphrase-multilingual-mpnet-base-v2 (768-dim, 한국어 포함 50개 언어)
    - 이미지: clip-ViT-B-32 (512-dim, 시각적 유사도 검색)

    asyncio.to_thread()로 이벤트 루프 블로킹 방지.
    """

    def __init__(
        self,
        text_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        image_model_name: str = "clip-ViT-B-32",
    ):
        # 텍스트 모델 (multilingual, 768-dim)
        logger.info("텍스트 임베딩 모델 로딩 중: %s", text_model_name)
        self._text_model = SentenceTransformer(text_model_name)
        logger.info("텍스트 임베딩 모델 로딩 완료")

        # 이미지 모델 (CLIP, 512-dim) — Windows safetensors mmap 오류 방지
        logger.info("이미지 임베딩 모델 로딩 중: %s", image_model_name)
        self._image_model = SentenceTransformer(
            image_model_name,
            model_kwargs={"use_safetensors": False},
        )
        logger.info("이미지 임베딩 모델 로딩 완료")

    # ──────────────────────────── 텍스트 임베딩 (768-dim) ────────────────────────────

    def _encode_text_sync(self, text: str) -> list[float]:
        embedding = self._text_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    async def encode_text(self, text: str) -> list[float]:
        """텍스트를 768차원 multilingual 벡터로 변환 (비동기)"""
        return await asyncio.to_thread(self._encode_text_sync, text)

    # ──────────────────────────── 이미지 임베딩 (512-dim) ────────────────────────────

    def _encode_image_sync(self, image: Image.Image) -> list[float]:
        embedding = self._image_model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    async def encode_image_from_bytes(self, image_bytes: bytes) -> list[float]:
        """이미지 바이트를 512차원 CLIP 벡터로 변환 (비동기)"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return await asyncio.to_thread(self._encode_image_sync, image)

    async def encode_image_from_url(
        self,
        url: str,
        http_client: httpx.AsyncClient,
    ) -> list[float] | None:
        """
        URL에서 이미지를 다운로드하여 벡터로 변환.
        실패 시 None 반환 (graceful degradation).
        """
        try:
            response = await http_client.get(url, timeout=15.0, follow_redirects=True)
            response.raise_for_status()
            return await self.encode_image_from_bytes(response.content)
        except Exception as e:
            logger.debug("이미지 임베딩 실패 (URL: %s): %s", url, e)
            return None
