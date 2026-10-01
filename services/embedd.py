from sentence_transformers import SentenceTransformer

from config import settings
from resources.prompts import EMBED_INSTRUCTION

# loaded once at import time
_model = SentenceTransformer(settings.embed_model)


def embed_chunks(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return _model.encode(
        EMBED_INSTRUCTION + query, normalize_embeddings=True
    ).tolist()
