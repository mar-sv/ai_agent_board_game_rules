
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter, HTTPException, Query
import uvicorn

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from rag.rag_oop import RAGService, ChatResponse
from boardgame_agents.rag.db_utils import search_available_games, print_available_tables

router = APIRouter(
    prefix="/boardgame_rag",
    tags=["Dashboard"],
    responses={404: {"description": "Not found"}},
)


rag_service: RAGService | None = None


class AddGameRequest(BaseModel):
    game_name: str
    session_id: str


class ChatRequest(BaseModel):
    user_input: str
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    rag_service = RAGService()

    yield
    pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.get("/chat", response_model=ChatResponse)
def chat_endpoint(
    user_input: str = Query(..., min_length=1),
    session_id: str = Query(...),
) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    answer = rag_service.chat(
        user_input=user_input,
        session_id=session_id,
    )

    return ChatResponse(answer=answer)


@router.post("/add_game_to_context")
def add_game_to_context_endpoint(payload: AddGameRequest):
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    rag_service.init_session_to_database(
        game_name=payload.game_name,
        session_id=payload.session_id,
    )


@router.get("/games/search")
def search_games(q: str = Query(..., min_length=2, description="Search query")):
    q = q.strip()
    if len(q) < 2:
        return {"games": []}

    try:
        #print_available_tables()
        results = search_available_games(q)
        return {"games": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Failed to search games") from e


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
