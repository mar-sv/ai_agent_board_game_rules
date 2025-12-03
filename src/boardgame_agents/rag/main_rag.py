from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_history_aware_retriever
from src.boardgame_agents.rag.prompt_templates_rag import get_history_aware_message, get_qa_message
from src.boardgame_agents.rag.rag_helpers import extend_chathistory, get_reranked_retriever
import os
from langchain_openai import ChatOpenAI


def call_rag():

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )
    retriever = get_reranked_retriever()

    context_q_prompt = get_history_aware_message()
    qa_prompt = get_qa_message(add_context=True)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_q_prompt)
    document_prompt = PromptTemplate.from_template(
        "From {source} (page {page}):\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_prompt=document_prompt,
        document_separator="\n\n---\n\n"
    )
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)

    chat_history = []

    print("Type 'exit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history})
        answer = response["answer"]
        print(answer)

        chat_history = extend_chathistory(chat_history, user_input, answer)


if __name__ == "__main__":
    call_rag()
