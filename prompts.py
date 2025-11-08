from typing import Dict, Any


class PromptTemplates:
    """Container for all prompt templates used in the research assistant."""

    @staticmethod
    def board_game_prompt_system() -> str:
        """System prompt for analyzing the PDF, and evaluate whether it is the correct board game search results."""

        return f"""You are an expert in board games. Your task is to evaluate whether a given PDF’s text corresponds to the official rules of a specific board game.

        Instructions:
        - Read the provided text carefully.
        - Decide if the text represents the official rules for that game.

        - After your label, provide a short (1–2 sentence) explanation for your reasoning."""
        # - Output exactly one of the following:
        #     - MATCH — if it clearly contains or is the official rules for [GAME NAME].
        #     - NOT_MATCH — if it is not, or if it’s unclear.

    @staticmethod
    def board_game_prompt_user(board_game: str, pdf_text: str) -> str:
        """User prompt for analyzing Google search results."""
        return f"""Target game: {board_game}

PDF text: {pdf_text}

Please analyze the text and evaluate whether the text is the board game rules to the target game"""


def create_message_pair(system_prompt: str, user_prompt: str) -> list[Dict[str, Any]]:
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
