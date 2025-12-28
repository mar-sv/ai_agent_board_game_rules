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
            chat_history JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
    """)


def insert_game_to_db(game_name, session_id, chat_history, table_name: str = "chat_history"):
    # Accept either uuid.UUID or a UUID string
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

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
    return {
        "game_name": game_name,
        "chat_history": chat_history  # already a dict if JSONB
    }
