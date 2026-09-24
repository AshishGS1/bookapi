from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

from config import settings

# same tokenizer the embedding model uses
_tokenizer = AutoTokenizer.from_pretrained(settings.embed_model)


@dataclass
class Chunk:
    text: str
    chunk_index: int


def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _tail_by_tokens(text: str, target_tokens: int) -> str:
    #retrieves overlap
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        tail.insert(0, word)
        if _token_len(" ".join(tail)) >= target_tokens:
            break
    return " ".join(tail)


def chunk_text(text: str) -> list[Chunk]:
    #splits text into chunks with overlap, based on token count
    tsize = settings.chunk_size
    toverlap = settings.chunk_overlap

    paras = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[Chunk] = []
    current = ""

    def flush(piece: str) -> None:
        if piece.strip():
            chunks.append(Chunk(text=piece.strip(), chunk_index=len(chunks)))

    for para in paras:
        if _token_len(para) > tsize:
            # split a single paragraph into multiple chunks if it's too big for one
            overlap = ""
            if current:
                flush(current)
                overlap = _tail_by_tokens(current, toverlap)

            words = para.split()
            piece = overlap
            for word in words:
                temp = f"{piece} {word}".strip()
                if _token_len(temp) > tsize:
                    flush(piece)
                    overlap = _tail_by_tokens(piece, toverlap)
                    piece = f"{overlap} {word}".strip()
                else:
                    piece = temp
            current = piece
            continue

        temp = f"{current}\n{para}".strip()
        if _token_len(temp) <= tsize:
            current = temp
        else:
            flush(current)
            overlap = _tail_by_tokens(current, toverlap)
            current = f"{overlap}\n{para}".strip()

    flush(current)
    return chunks