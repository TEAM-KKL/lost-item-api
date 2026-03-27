"""FastAPI 의존성 주입 함수들"""

from fastapi import Request

from app.services.embedding import EmbeddingService
from app.services.ingest import IngestService
from app.services.police_client import PoliceAPIClient
from app.services.vector_store import VectorStoreService


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding


def get_vector_store(request: Request) -> VectorStoreService:
    return request.app.state.vector_store


def get_police_client(request: Request) -> PoliceAPIClient:
    return request.app.state.police_client


def get_ingest_service(request: Request) -> IngestService:
    return request.app.state.ingest_service
