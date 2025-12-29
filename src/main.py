from src.boardgame_agents.rag.rag_oop import RAGService, ChatResponse
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

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
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized"
        )

    answer = rag_service.chat(
        user_input=payload.user_input,
        session_id=payload.session_id
    )

    return ChatResponse(answer=answer)


@router.post("/add_game")
def add_game_to_context_endpoint(payload: AddGameRequest):
    if rag_service is None:
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    rag_service.init_session_to_database(
        game_name=payload.game_name,
        session_id=payload.session_id,
    )
    # return ChatResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8080, reload=True)
