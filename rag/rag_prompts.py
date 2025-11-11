from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


class PromptTemplates:
    """Container for all prompt templates used in the research assistant."""

    @staticmethod
    def history_aware_sys_prompt() -> str:
        """System prompt for providing context to the rag"""

        return f"""You are a history-aware query rewriter for a retrieval-augmented system about board games.

Your job: given the current user message and the prior conversation, rewrite the user message into a single, complete search query that:
- is specific to board games,
- keeps the user’s intent,
- adds missing context from the conversation (game name, variant, edition, expansion, phase/step of the turn),
- and is suitable for searching over short rule chunks.

Use only information that actually appears in the conversation. Do NOT invent game names or expansions. If the user is asking about “that action” or “this step”, resolve what “that” or “this” refers to from the previous turns.

If the user is clearly talking about rules, turns, actions, setup, victory conditions, or component limits, make sure the rewritten query mentions the exact game title and the rule area.

Output ONLY the rewritten query, nothing else."""

    @staticmethod
    def answer_sys_prompt() -> str:
        """System prompt for providing context to the rag"""

        return f"""You are an expert assistant that answers questions about board games using retrieved rule chunks and metadata.

Your task:
- Read the retrieved context carefully and answer the user’s question as accurately and concretely as possible.
- Base your answer ONLY on the provided context (rule chunks and metadata). Do not make up or assume rules that are not supported by the retrieved text.
- When the context includes multiple rule chunks, synthesize them into a clear, concise explanation.
- Always specify the relevant game name and rule section if the context allows.
- If the retrieved information is incomplete or conflicting, say so explicitly and summarize what can and cannot be inferred.
- If the user’s question cannot be answered from the given context, respond with: “The rules provided don’t specify this.”

Keep responses short, structured, and clear for easy reference during gameplay"""

    @staticmethod
    def board_game_prompt_user(input: str) -> str:
        """User prompt for analyzing Google search results."""
        return f""""User asks about board game rules: {input}"""


def create_chat_prompts(system_prompt: str, user_prompt: str):
    return ChatPromptTemplate.from_messages([("system", system_prompt),
                                             (MessagesPlaceholder(
                                                 "chat_history")),
                                             ("human", user_prompt)])


def create_message_pair(system_prompt: str, chat_history: str, user_prompt: str) -> list[Dict[str, Any]]:
    """
    Create a standardized message pair for LLM interactions.

    Args:
        system_prompt: The system message content
        user_prompt: The user message content

    Returns:
        List containing system and user message dictionaries
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# Convenience functions for creating complete message arrays
def get_rules_evaluation_message(
    board_game: str, pdf_text: str
) -> list[Dict[str, Any]]:
    """Get messages for Google results analysis."""
    return create_message_pair(
        PromptTemplates.board_game_prompt_system(),
        PromptTemplates.board_game_prompt_user(board_game, pdf_text),
    )
