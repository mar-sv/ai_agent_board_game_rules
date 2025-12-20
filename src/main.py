from src.boardgame_agents.rag.rag_oop import RAGService, ChatResponse
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from contextlib import asynccontextmanager

router = APIRouter(
    prefix="/boardgame-rag",
    tags=["Dashboard"],
    responses={404: {"description": "Not found"}},
)

rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    rag_service = RAGService()

    yield
    pass


app = FastAPI(lifespan=lifespan)


@router.post("/chat")
def chat_endpoint(user_input: str) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    answer = rag_service.chat(user_id=None, user_input=user_input)
    return ChatResponse(answer=answer)


@router.post("/add_game")
def add_game_to_context_endpoint(user_input: str) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    answer = rag_service.add_game_to_context(
        user_id=None, user_input=user_input)
    return ChatResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8080, reload=True)
