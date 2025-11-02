from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import pdfplumber
from web_crawler import query_google
from prompts import get_rules_evaluation_message
load_env = load_dotenv()

llm = init_chat_model("gpt-4o-mini")


class State(TypedDict):
    # messages: Annotated[List, add_messages]
    game_name: str | None
    pdf_text: str | None
    llm_evaluation: str | None


def google_search(state):
    game_name = state.get("game_name", "")
    pdf_text = query_google(game_name)
    return {"pdf_text": pdf_text}


def analyze_pdf(state):
    game_name = state.get("game_name", "")
    pdf_text = state.get("pdf_text", "")
    messages = get_rules_evaluation_message(game_name, pdf_text)
    reply = llm.invoke(messages)

    return {"llm_evaluation": reply.content}


graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("analyze_pdf", analyze_pdf)

graph_builder.add_edge(START, end_key="google_search")
graph_builder.add_edge(start_key="google_search", end_key="analyze_pdf")

graph_builder.add_edge(start_key="analyze_pdf", end_key=END)

graph = graph_builder.compile()


def run_chatbot():
    # print("Multi-Source Research Agent")
    # print("Type 'exit' to quit\n")

    # while True:
    #     user_input = input("Ask me anything: ")
    #     if user_input.lower() == "exit":
    #         print("Bye")
    #         break
    state = {"game_name": "catan",
             "pdf_text": None,
             "llm_evaluation": None}

    # print("\nStarting parallel research process...")
    # print("Launching Google, Bing, and Reddit searches...\n")
    final_state = graph.invoke(state)

    llm_evaluation = final_state.get("llm_evaluation", "")

    if llm_evaluation:
        print(f"\nFinal Answer:\n{llm_evaluation}\n")

    if llm_evaluation.split(" — ")[0] == "MATCH":
        pass
        # here do stuff. Chunk the pdf text and add it to db

    print("-" * 80)


if __name__ == "__main__":
    run_chatbot()
