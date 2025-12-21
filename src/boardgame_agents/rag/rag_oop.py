from pydantic import BaseModel
from typing import Dict, List, Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_history_aware_retriever

from src.boardgame_agents.rag.prompt_templates_rag import (
    get_history_aware_message,
    get_qa_message,
)
from src.boardgame_agents.rag.rag_helpers import extend_chathistory, get_reranked_retriever, get_llm_model

# ---------- Pydantic models ----------


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class RAGService:
    def __init__(self) -> None:
        self.llm = get_llm_model()
        self.retriever = get_reranked_retriever()

    def insert_game_to_database(game_name, session_id):
        pass

    def add_game_to_context(self, game_name: str):
        context_q_prompt = get_history_aware_message()
        qa_prompt = get_qa_message(game_name, add_context=True)

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, context_q_prompt
        )

        document_prompt = PromptTemplate.from_template(
            "From {source} (page {page}):\n{page_content}"
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt,
            document_prompt=document_prompt,
            document_separator="\n\n---\n\n",
        )

        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            question_answer_chain,
        )

    def _get_history_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        return self.chat_histories.get(user_id, [])

    def _set_history_for_user(self, user_id: str, history: List[Dict[str, Any]]) -> None:
        # change later to insert to db
        self.chat_histories[user_id] = history

    def chat(self, user_id: str, user_input: str) -> str:
        # chat_history = self._get_history_for_user(user_id)
        chat_history = []

        response = self.rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history}
        )
        answer = response["answer"]

        new_history = extend_chathistory(chat_history, user_input, answer)
        # self._set_history_for_user(user_id, new_history)

        return answer
