"""The API contract. The browser never imports model or training code."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=600)
    genre: Literal["Indie pop", "Pop", "Hip-hop", "R&B", "Rock", "Acoustic", "Country"] = "Indie pop"
    mood: Literal["Dreamy", "Melancholic", "Uplifting", "Romantic", "Bold"] = "Dreamy"
    length: Literal["short", "standard", "long"] = "standard"
    creativity: int = Field(default=50, ge=0, le=100)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Give your song a little starting point.")
        return value

    def model_prompt(self) -> str:
        # The UI controls become conditioning text rather than unsupported model arguments.
        scope = {"short": "a short hook", "standard": "a verse and chorus", "long": "a longer lyric draft"}[self.length]
        return f"Write {scope} for a {self.mood.lower()} {self.genre} song. {self.prompt}"


def stage_for(progress: float) -> str:
    if progress < 34:
        return "Finding the feeling"
    if progress < 75:
        return "Putting words into motion"
    return "Finishing this little take"
