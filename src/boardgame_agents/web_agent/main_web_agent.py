import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from web_crawler import query_google
from prompts_templates_web import get_rules_evaluation_message, BoardGameEvaluation
from db_insertion import process_and_insert_pdf, document_exists_sql
load_env = load_dotenv()

llm = init_chat_model("gpt-4o-mini")


class State(TypedDict):
    # messages: Annotated[List, add_messages]
    game_name: str | None
    pdf_text: str | None
    structured_output: BoardGameEvaluation | None


def google_search(state):
    game_name = state.get("game_name", "")
    pdf_text = query_google(game_name)
    return {"pdf_text": pdf_text}


def analyze_pdf(state):
    game_name = state.get("game_name", "")
    pdf_text = state.get("pdf_text", "")
    messages = get_rules_evaluation_message(game_name, pdf_text)
    structured_llm = llm.with_structured_output(BoardGameEvaluation)
    # reply = llm.invoke(messages)
    structured_output = structured_llm.invoke(messages)

    return {"structured_output": structured_output}


graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("analyze_pdf", analyze_pdf)

graph_builder.add_edge(START, end_key="google_search")
graph_builder.add_edge(start_key="google_search", end_key="analyze_pdf")

graph_builder.add_edge(start_key="analyze_pdf", end_key=END)

graph = graph_builder.compile()


def run_chatbot(csv_name, board_game_name_column):
    game_names = pd.read_csv(csv_name)[board_game_name_column].to_list()
    for game_name in game_names:
        state = {"game_name": game_name,
                 "pdf_text": None,
                 "boardgame_evaluation": None}

        if document_exists_sql(game_name):
            print(f"{game_name} already exists, skipping...")
            continue

        final_state = graph.invoke(state)

        if structured_output := final_state.get("structured_output", ""):
            pass

        if structured_output.rules:
            pdf_path = f"pdfs/{final_state.get('game_name')}.pdf"

            process_and_insert_pdf(
                pdf_path=pdf_path, creator=structured_output.creator)
        print("-" * 80)


if __name__ == "__main__":
    csv_name = r"C:\board_game_rag\rag_test.csv"
    board_game_name_column = "board_game_name"
    run_chatbot(csv_name=csv_name,
                board_game_name_column=board_game_name_column)
