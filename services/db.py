import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from models import SrcChunk

from config import settings

client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api)


def ensure_collection() -> None:
    """Creates the collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in existing:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.dims, distance=Distance.COSINE
            ),
        )


def _point_id(doc_id: str, chunk_index: int) -> str:
    # deterministic id — re-ingesting the same doc overwrites its old points
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode()).hexdigest()


def upsert_chunks(doc_id: str, texts: list[str], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=_point_id(doc_id, i),
            vector=vector,
            payload={"text": text, "doc_id": doc_id, "chunk_index": i},
        )
        for i, (text, vector) in enumerate(zip(texts, vectors))
    ]
    client.upsert(collection_name=settings.collection_name, points=points)


def search(query_vector: list[float], top_k: int) -> list[dict]:
    result = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        SrcChunk(
            text= point.payload["text"],
            score= point.score,
            chunk_index= point.payload["chunk_index"],
            doc_id= point.payload["doc_id"],
        )
        for point in result.points
    ] #use model here
