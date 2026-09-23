from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import book
from services import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # make sure the collection exists
    db.ensure_collection()
    yield


app = FastAPI(title="BookAPI", lifespan=lifespan)

app.include_router(book.router, prefix="/book", tags=["rag"])


@app.get("/health")
async def health():
    return {"status": "ok"}