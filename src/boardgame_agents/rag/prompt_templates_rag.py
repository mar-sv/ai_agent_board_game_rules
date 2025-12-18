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
    def answer_sys_prompt(game_name) -> str:
        """System prompt for providing context to the rag"""

        return f"""You are an expert assistant that answers questions about board games using retrieved rule chunks and metadata. 
        You are an expert in the game {game_name}

Your task:
- Read the retrieved context carefully and answer the user’s question as accurately and concretely as possible.
- Base your answer ONLY on the provided context (rule chunks and metadata). Do not make up or assume rules that are not supported by the retrieved text.
- When the context includes multiple rule chunks, synthesize them into a clear, concise explanation.
- Always specify the relevant game name and rule section if the context allows. Include e.g. page number if available.
- If the retrieved information is incomplete or conflicting, say so explicitly and summarize what can and cannot be inferred.
- If the user’s question cannot be answered from the given context, respond with: “The rules provided don’t specify this.”

Keep responses short, structured, and clear for easy reference during gameplay"""

    @staticmethod
    def board_game_prompt_user() -> str:
        """User prompt for analyzing Google search results."""
        return """User asks about board game rules: {input}"""


def create_chat_prompts(system_prompt: str, user_prompt: str, add_context: bool = False):
    messages = [("system", system_prompt)]

    if add_context:
        messages.append(("system", "Context: {context}"))

    messages.append(MessagesPlaceholder("chat_history"))
    messages.append(("human", user_prompt))

    return ChatPromptTemplate.from_messages(messages)


# Convenience functions for creating complete message arrays
def get_history_aware_message():
    """Get messages for Google results analysis."""
    return create_chat_prompts(
        PromptTemplates.history_aware_sys_prompt(),
        PromptTemplates.board_game_prompt_user(),
    )


def get_qa_message(game_name, add_context=True):
    """Get messages for Google results analysis."""
    return create_chat_prompts(
        PromptTemplates.answer_sys_prompt(game_name),
        PromptTemplates.board_game_prompt_user(),
        add_context=add_context
    )
