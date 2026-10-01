from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import book
from services import db
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # make sure the collection exists
    db.ensure_collection()
    yield


app = FastAPI(title="BookAPI", lifespan=lifespan)

# Add cors middleware to allow requests from other origins(frontend).
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(book.router, prefix="/book", tags=["rag"])


@app.get("/health")
async def health():
    return {"status": "ok"}