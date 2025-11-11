import rag_functions as rf
from rag_prompts import get_history_aware_message, get_qa_message
from langchain_classic.chains import create_history_aware_retriever
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_retrival_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

if __name__ == "__main__":

    llm = init_chat_model("gpt-4o-mini")
    # History aware

    user_input = "When do i move the robber?"
    chat_history = []
    # create history aware message
    context_q_prompt = get_history_aware_message(user_input)
    retriver = rf.get_retirver()

    history_aware_retriver = create_history_aware_retriever(
        llm, retriver, context_q_prompt)

    history_aware_retriver.invoke(
        {"input": user_input, "chat_history": chat_history})

    # create full rag
    qa_prompt = get_qa_message(user_input)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrival_chain(
        history_aware_retriver, question_answer_chain)

    rag_chain.invoke({"input": user_input, "chat_history": chat_history})
