from sentence_transformers import SentenceTransformer

from config import settings

# loaded once at import time
_model = SentenceTransformer(settings.embed_model)

# query prefix for bge model
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embed_chunks(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return _model.encode(
        _QUERY_INSTRUCTION + query, normalize_embeddings=True
    ).tolist()
