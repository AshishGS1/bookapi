from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer

from config import settings

# Use the same tokenizer as the embedding model so chunk sizes match its token limits.
_tokenizer = AutoTokenizer.from_pretrained(settings.embed_model)


@dataclass
class Chunk:
    text: str
    chunk_index: int


def _token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _tail_by_tokens(text: str, target_tokens: int) -> str:
    """Return the end of the text to carry into the next chunk."""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        tail.insert(0, word)
        if _token_len(" ".join(tail)) >= target_tokens:
            break
    return " ".join(tail)


def chunk_text(text: str) -> list[Chunk]:
    """Group paragraphs into token-limited chunks while preserving context."""
    tsize = settings.chunk_size
    toverlap = settings.chunk_overlap

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[Chunk] = []
    current = ""

    def flush(piece: str) -> None:
        if piece.strip():
            chunks.append(Chunk(text=piece.strip(), chunk_index=len(chunks)))

    for para in paragraphs:
        if _token_len(para) > tsize:
            # Split long paragraphs at word boundaries when they do not fit.
            words = para.split()
            piece = ""
            for word in words:
                trial = f"{piece} {word}".strip()
                if _token_len(trial) > tsize:
                    flush(piece)
                    piece = word
                else:
                    piece = trial
            current = piece
            continue

        trial = f"{current}\n{para}".strip()
        if _token_len(trial) <= tsize:
            current = trial
        else:
            flush(current)
            overlap = _tail_by_tokens(current, toverlap)
            current = f"{overlap}\n{para}".strip()

    flush(current)
    return chunks
