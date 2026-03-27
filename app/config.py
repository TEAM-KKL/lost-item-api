from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    anthropic_api_key: str = ""

    # OpenAI (legacy — 미사용)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # 경찰청 공공데이터포털 API
    police_api_key: str = ""
    police_api_base_url: str = (
        "https://apis.data.go.kr/1320000/LosfundInfoInqireService"
        "/getLosfundInfoAccToClAreaPd"
    )

    # MongoDB (크롤링 데이터)
    mongo_uri: str = "mongodb://localhost:27017"
    downloads_dir: str = "E:/projects/lostitem/downloads"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "lost_items"

    # 앱
    app_env: str = "development"
    log_level: str = "INFO"

    # 임베딩 모델
    clip_model_name: str = "clip-ViT-B-32"                             # 이미지 모델
    text_model_name: str = "paraphrase-multilingual-mpnet-base-v2"     # 텍스트 모델 (한국어 지원)
    text_embedding_dim: int = 768
    image_embedding_dim: int = 512


@lru_cache
def get_settings() -> Settings:
    return Settings()
