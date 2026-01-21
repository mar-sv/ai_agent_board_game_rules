from __future__ import annotations
from typing import List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv
import uuid
import json
import psycopg2
from psycopg2.extras import Json

load_dotenv()
PG_DSN = os.getenv("DB_DSN", "")


def ensure_table(cur, table_name: str = "chat_history"):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            session_id UUID PRIMARY KEY,
            game_name  TEXT NOT NULL,
            chat_history JSONB NOT NULL DEFAULT '[]'::jsonb
        );
    """)


def insert_game_to_db(game_name, session_id, chat_history, table_name: str = "chat_history"):
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    chat_history = [] if chat_history is None else lc_to_db_json(chat_history)
    if not isinstance(chat_history, list):
        raise TypeError(
            f"chat_history must be a list, got {type(chat_history)}")

    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            ensure_table(cur, table_name)

            cur.execute(f"""
                INSERT INTO {table_name} (session_id, game_name, chat_history)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET game_name = EXCLUDED.game_name,
                    chat_history = EXCLUDED.chat_history;
            """, (session_id, game_name, Json(chat_history)))

    print("Inserted/updated chat history row.")


def get_game_and_chat_history(session_id, table_name: str = "chat_history"):
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT game_name, chat_history
                FROM {table_name}
                WHERE session_id = %s;
            """, (session_id,))

            row = cur.fetchone()

    if row is None:
        return None  # or raise

    game_name, chat_history = row

    return game_name, db_json_to_lc(chat_history)


def db_json_to_lc(history: Optional[List[Dict[str, Any]]]) -> List[BaseMessage]:
    history = history or []
    out: List[BaseMessage] = []

    for m in history:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")

        if role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            # ToolMessage typically needs a tool_call_id; keep optional for your schema
            out.append(ToolMessage(content=content,
                       tool_call_id=m.get("tool_call_id", "tool")))
        else:
            # fallback: treat unknown as user text
            out.append(HumanMessage(content=content))

    return out

# ---------- LangChain messages -> DB JSON ----------


def lc_to_db_json(messages: Optional[List[BaseMessage]]) -> List[Dict[str, Any]]:
    messages = messages or []
    out: List[Dict[str, Any]] = []

    for msg in messages:
        # msg.type is typically: "human", "ai", "system", "tool"
        t = getattr(msg, "type", None)

        if t in ("human", "user"):
            role = "user"
        elif t in ("ai", "assistant"):
            role = "assistant"
        elif t == "system":
            role = "system"
        elif t == "tool":
            role = "tool"
        else:
            role = "user"

        item: Dict[str, Any] = {"role": role, "content": msg.content}

        # preserve tool_call_id if present
        if role == "tool" and hasattr(msg, "tool_call_id"):
            item["tool_call_id"] = getattr(msg, "tool_call_id")

        out.append(item)

    return out


def print_available_tables():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()

    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    """)

    for schema, table in cur.fetchall():
        print(f"{schema}.{table}")

    cur.close()
    conn.close()


def search_available_games(query: str, limit: int = 20) -> List[Dict[str, str]]:
    q = (query or "").strip()
    if len(q) < 2:
        return []

    sql = """
        SELECT
            game_name
        FROM available_games
        WHERE game_name ILIKE %s
        ORDER BY
            CASE
                WHEN game_name ILIKE %s THEN 0  -- prefix match first
                ELSE 1
            END,
            game_name ASC
        LIMIT %s;
    """

    like_any = f"%{q}%"
    like_prefix = f"{q}%"

    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (like_any, like_prefix, limit))
            rows = cur.fetchall()

    return [{"title": row[0]} for row in rows]
