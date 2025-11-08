from pydantic import BaseModel, Field


class BoardGameEvaluation(BaseModel):
    # boardgame_name: str = Field("Name of the boardgame")
    rules: bool = Field("Evaluation whether the pdf is the board game or not")
