from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import pdfplumber

from web_crawler import query_google

load_env = load_dotenv()

llm = init_chat_model("gpt-4o-mini")


class State(TypedDict):
    # messages: Annotated[List, add_messages]
    url_basename: str | None
    game_name: str | None


def google_search(state):
    # query_google()
    # return
    pass


def analyze_pdf(state):
    # return
    pass


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages).replace("\n", " ")


graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("analyze_pdf", analyze_pdf)

graph_builder.add_edge(START, start_key="google_search")
graph_builder.add_edge(start_key="google_search", end_key="analyze_pdf")

graph_builder.add_edge(start_key="analyze_pdf", end_key=END)
