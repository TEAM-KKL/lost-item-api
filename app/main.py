"""FastAPI 애플리케이션 - 경찰청 습득물 AI 검색 시스템"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api.routes import health, ingest, mongo_ingest, search
from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.ingest import IngestService
from app.services.mongo_client import MongoLostItemClient
from app.services.mongo_ingest import MongoIngestService
from app.services.police_client import PoliceAPIClient
from app.services.vector_store import VectorStoreService

# ──────────────────────────── 로깅 설정 ────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ──────────────────────────── Lifespan ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    앱 시작/종료 시 실행되는 lifespan context manager.

    시작:
    1. CLIP 모델 로드 (최초 실행 시 ~350MB 다운로드)
    2. Qdrant 비동기 클라이언트 초기화
    3. httpx 비동기 클라이언트 초기화
    4. 서비스 인스턴스 app.state에 등록
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=== 경찰청 습득물 AI 검색 시스템 시작 중 ===")

    # 1. 임베딩 모델 초기화 (텍스트: multilingual-mpnet / 이미지: CLIP)
    logger.info(
        "임베딩 모델 초기화 중... 텍스트=%s, 이미지=%s",
        settings.text_model_name,
        settings.clip_model_name,
    )
    embedding_service = EmbeddingService(
        text_model_name=settings.text_model_name,
        image_model_name=settings.clip_model_name,
    )

    # 2. Qdrant 클라이언트
    logger.info("Qdrant 클라이언트 초기화: %s:%d", settings.qdrant_host, settings.qdrant_port)
    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    vector_store = VectorStoreService(qdrant_client)

    # 2-1. Qdrant 컬렉션 자동 생성 (없으면 생성, 있으면 스킵)
    await vector_store.ensure_collection()

    # 3. httpx 클라이언트
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    # 4. 경찰청 API 클라이언트
    police_client = PoliceAPIClient(
        http_client=http_client,
        api_key=settings.police_api_key,
        base_url=settings.police_api_base_url,
    )

    # 5. 경찰청 API 인제스트 서비스
    ingest_service = IngestService(
        police_client=police_client,
        embedding_service=embedding_service,
        vector_store=vector_store,
        http_client=http_client,
    )

    # 6. MongoDB 클라이언트 + 인제스트 서비스
    logger.info("MongoDB 클라이언트 초기화: %s", settings.mongo_uri)
    mongo_client = MongoLostItemClient(mongo_uri=settings.mongo_uri)
    mongo_ingest_service = MongoIngestService(
        mongo_client=mongo_client,
        embedding_service=embedding_service,
        vector_store=vector_store,
        downloads_dir=settings.downloads_dir,
    )

    # app.state에 등록 (deps.py에서 주입)
    app.state.embedding = embedding_service
    app.state.qdrant_client = qdrant_client
    app.state.vector_store = vector_store
    app.state.http_client = http_client
    app.state.police_client = police_client
    app.state.ingest_service = ingest_service
    app.state.mongo_client = mongo_client
    app.state.mongo_ingest_service = mongo_ingest_service

    # OpenAI API 키 환경변수 설정 (PydanticAI 에이전트용)
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    logger.info("=== 모든 서비스 초기화 완료. API 서버 준비됨 ===")
    logger.info("Swagger UI: http://localhost:8000/docs")
    logger.info("Qdrant Dashboard: http://localhost:6333/dashboard")

    yield  # 앱 실행

    # 종료 시 클린업
    logger.info("서버 종료 중...")
    await http_client.aclose()
    await qdrant_client.close()
    await mongo_client.close()
    logger.info("서버 종료 완료")


# ──────────────────────────── FastAPI 앱 ────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="경찰청 습득물 AI 검색 시스템",
        description="""
## 개요
경찰청 공공데이터의 습득물 정보를 CLIP 모델로 벡터 임베딩하여,
**텍스트** 또는 **이미지**로 분실물을 찾아주는 AI 검색 시스템입니다.

## 주요 기능
- 📝 **텍스트 검색**: "검정 가죽 지갑 잃어버렸어요" → 유사 습득물 반환
- 🖼️ **이미지 검색**: 잃어버린 물건 사진 업로드 → 시각적으로 유사한 습득물 반환
- 🔀 **복합 검색**: 텍스트 + 이미지 동시 활용
- 🤖 **AI 에이전트**: GPT-4o가 쿼리를 분석해 최적화된 검색 수행

## 기술 스택
- **임베딩**: CLIP (clip-ViT-B-32) - 텍스트와 이미지를 동일 512차원 공간에 임베딩
- **벡터 DB**: Qdrant - named vectors로 text/image 벡터 동시 관리
- **AI 에이전트**: PydanticAI + OpenAI GPT-4o
        """.strip(),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS (개발 환경)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(mongo_ingest.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
