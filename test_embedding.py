"""임베딩 서비스 동작 테스트 - ASCII 출력"""
import os
import asyncio
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TQDM_DISABLE"] = "1"   # tqdm stdout 오류 방지

import transformers
transformers.logging.set_verbosity_error()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.embedding import EmbeddingService


async def main():
    print("=== EmbeddingService Test ===")
    svc = EmbeddingService()

    # Text embedding
    v = await svc.encode_text("black leather wallet")
    print(f"Text vector dim: {len(v)}")
    print(f"First 3 values: {[round(x, 4) for x in v[:3]]}")
    assert len(v) == 512, f"Expected 512 dims, got {len(v)}"

    # Image embedding
    import io
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    iv = await svc.encode_image_from_bytes(img_bytes)
    print(f"Image vector dim: {len(iv)}")
    assert len(iv) == 512, f"Expected 512 dims, got {len(iv)}"

    # Combined vector
    combined = EmbeddingService.combine_vectors(v, iv, text_weight=0.6)
    print(f"Combined vector dim: {len(combined)}")
    assert len(combined) == 512

    import numpy as np
    norm = round(float(np.linalg.norm(combined)), 4)
    print(f"Combined vector norm (should be ~1.0): {norm}")

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
