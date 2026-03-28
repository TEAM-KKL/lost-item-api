"""FastAPI application entrypoint."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api.routes import health, images, ingest, mongo_ingest, search
from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.ingest import IngestService
from app.services.mongo_client import MongoLostItemClient
from app.services.mongo_ingest import MongoIngestService
from app.services.police_client import PoliceAPIClient
from app.services.search_session import SearchSessionService
from app.services.vector_store import VectorStoreService


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=== 경찰청 습득물 AI 검색 서비스 시작 ===")

    embedding_service = EmbeddingService(
        text_model_name=settings.text_model_name,
        image_model_name=settings.clip_model_name,
    )

    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    vector_store = VectorStoreService(qdrant_client)
    await vector_store.ensure_collection()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    police_client = PoliceAPIClient(
        http_client=http_client,
        api_key=settings.police_api_key,
        base_url=settings.police_api_base_url,
    )
    ingest_service = IngestService(
        police_client=police_client,
        embedding_service=embedding_service,
        vector_store=vector_store,
        http_client=http_client,
    )

    mongo_client = MongoLostItemClient(mongo_uri=settings.mongo_uri)
    mongo_ingest_service = MongoIngestService(
        mongo_client=mongo_client,
        embedding_service=embedding_service,
        vector_store=vector_store,
        downloads_dir=settings.downloads_dir,
    )
    search_session_service = SearchSessionService(mongo_uri=settings.mongo_uri)
    await search_session_service.ensure_indexes()

    app.state.embedding = embedding_service
    app.state.qdrant_client = qdrant_client
    app.state.vector_store = vector_store
    app.state.http_client = http_client
    app.state.police_client = police_client
    app.state.ingest_service = ingest_service
    app.state.mongo_client = mongo_client
    app.state.mongo_ingest_service = mongo_ingest_service
    app.state.search_session_service = search_session_service

    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    logger.info("Swagger UI: http://localhost:8000/docs")
    logger.info("Qdrant Dashboard: http://localhost:6333/dashboard")
    yield

    logger.info("서버 종료 중...")
    await http_client.aclose()
    await qdrant_client.close()
    await mongo_client.close()
    await search_session_service.close()
    logger.info("서버 종료 완료")


def create_app() -> FastAPI:
    app = FastAPI(
        title="경찰청 습득물 AI 검색 서비스",
        description=(
            "경찰청 습득물 데이터를 벡터 검색과 AI 에이전트로 탐색하는 API입니다. "
            "텍스트 검색, 이미지 검색, 복합 검색, 세션 기반 검색 흐름을 지원합니다."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(images.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(mongo_ingest.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
