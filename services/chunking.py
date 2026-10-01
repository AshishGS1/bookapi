from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

from config import settings

from chonkie import SemanticChunker


# same tokenizer the embedding model uses
_tokenizer = AutoTokenizer.from_pretrained(settings.embed_model)


@dataclass
class Chunk:
    text: str
    chunk_index: int

_semantic_chunker = SemanticChunker(
    embedding_model=settings.embed_model,
    embedding_dim=settings.dims,
    threshold=0.5,
    chunk_size=settings.chunk_size,
    similarity_window=7,
)


def chunk_text(text: str) -> list[Chunk]:
    semantic_chunks = _semantic_chunker.chunk(text)

    return [
        Chunk(text=chunk.text.strip(), chunk_index=index)
        for index, chunk in enumerate(semantic_chunks)
        if chunk.text.strip()
    ]

# def _token_len(text: str) -> int:
#     return len(_tokenizer.encode(text, add_special_tokens=False))


# def _get_overlap(text: str, toverlap: int) -> str:
#     #retrieves overlap
#     words = text.split()
#     tail: list[str] = []
#     for word in reversed(words):
#         tail.insert(0, word)
#         if _token_len(" ".join(tail)) >= toverlap:
#             break
#     return " ".join(tail)


# def chunk_text(text: str) -> list[Chunk]:
#     #splits text into chunks with overlap, based on token count
#     tsize = settings.chunk_size
#     toverlap = settings.chunk_overlap

#     paras = [p.strip() for p in text.split("\n") if p.strip()]

#     chunks: list[Chunk] = []
#     current = ""

#     def flush(piece: str) -> None:
#         if piece.strip():
#             chunks.append(Chunk(text=piece.strip(), chunk_index=len(chunks)))

#     for para in paras:
#         if _token_len(para) > tsize:
#             # split a single paragraph into multiple chunks if it's too big for one
#             overlap = ""
#             if current:
#                 flush(current)
#                 overlap = _get_overlap(current, toverlap)

#             words = para.split()
#             piece = overlap
#             for word in words:
#                 temp = f"{piece} {word}".strip()
#                 if _token_len(temp) > tsize:
#                     flush(piece)
#                     overlap = _get_overlap(piece, toverlap)
#                     piece = f"{overlap} {word}".strip()
#                 else:
#                     piece = temp
#             current = piece
#             continue

#         temp = f"{current}\n{para}".strip()
#         if _token_len(temp) <= tsize:
#             current = temp
#         else:
#             flush(current)
#             overlap = _get_overlap(current, toverlap)
#             current = f"{overlap}\n{para}".strip()

#     flush(current)
#     return chunks