"""
Qdrant 컬렉션 초기 생성 스크립트.

사용법:
    uv run python scripts/init_qdrant.py            # 없으면 생성 (멱등)
    uv run python scripts/init_qdrant.py --recreate  # 기존 컬렉션 삭제 후 재생성

.env 파일의 QDRANT_HOST, QDRANT_PORT 설정을 읽어 컬렉션을 생성합니다.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.services.vector_store import VectorStoreService


async def main(recreate: bool = False) -> None:
    settings = get_settings()
    print(f"Qdrant 연결 중: {settings.qdrant_host}:{settings.qdrant_port}")

    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vector_store = VectorStoreService(client)

    try:
        if recreate:
            exists = await client.collection_exists(vector_store.COLLECTION)
            if exists:
                print(f"기존 컬렉션 '{vector_store.COLLECTION}' 삭제 중...")
                await client.delete_collection(vector_store.COLLECTION)
                print("삭제 완료")
            else:
                print(f"컬렉션 '{vector_store.COLLECTION}' 없음 → 바로 생성")

        await vector_store.ensure_collection()
        info = await vector_store.get_collection_info()
        print(f"\n컬렉션 '{vector_store.COLLECTION}' 상태:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("\n초기화 완료!")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant 컬렉션 초기화")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="기존 컬렉션을 삭제하고 재생성 (dim 변경 시 필요)",
    )
    args = parser.parse_args()
    asyncio.run(main(recreate=args.recreate))
