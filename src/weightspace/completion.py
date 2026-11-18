from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    text: str
    answer: str | None
    finished: bool
    tokens: int
    looped: bool = False
