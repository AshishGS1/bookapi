import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from auth import verify_auth_key
from config import settings
from models import UploadRes, QueryReq, QueryRes, SrcChunk
from services import chunking, embedd, extract, aipolish, db

router = APIRouter(dependencies=[Depends(verify_auth_key)])


@router.post("/upload", response_model=UploadRes)
async def upload_doc(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = Path(file.filename).stem
        text = extract.extract_text(tmp_path)
        chunks = chunking.chunk_text(text)
        vectors = embedd.embed_chunks([c.text for c in chunks])
        db.ensure_collection()
        db.upsert_chunks(doc_id, [c.text for c in chunks], vectors)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadRes(doc_id=doc_id, num_chunks=len(chunks))


@router.post("/query", response_model=QueryRes)
async def query_doc(payload: QueryReq):
    top_k = payload.top_k or settings.top_k
    query_vector = embedd.embed_query(payload.query)
    hits = db.search(query_vector, top_k)
    answer = aipolish.generate_answer(payload.query, [h["text"] for h in hits])
    return QueryRes(answer=answer, sources=[SrcChunk(**h) for h in hits])