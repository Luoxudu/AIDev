"""AIDev Web 前端 — FastAPI 入口。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.documents import router as documents_router
from api.memory import router as memory_router
from api.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.sessions import init_db
    from src.retrieval.reranker import check_and_download_reranker_model
    init_db()
    check_and_download_reranker_model()
    yield


app = FastAPI(title="AIDev", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(memory_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
