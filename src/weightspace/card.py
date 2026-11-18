import re
from typing import Literal

Status = Literal["draft", "frozen"]

HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
STATUS_LINE = re.compile(r"^Status: (\w+)$", re.MULTILINE)
UNFILLED_ALLOWED = {"Amendments"}


def sections(text: str) -> dict[str, str]:
    parts = HEADING.split(text)
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def status(text: str) -> Status:
    match = STATUS_LINE.search(text)
    if not match or match.group(1) not in ("draft", "frozen"):
        raise ValueError("card has no 'Status: draft' or 'Status: frozen' line")
    return match.group(1)


def unfilled(text: str, template: str) -> list[str]:
    filled = sections(text)
    return [
        name
        for name, body in sections(template).items()
        if name not in UNFILLED_ALLOWED and filled.get(name, body) == body
    ]


def frozen(text: str) -> str:
    return STATUS_LINE.sub("Status: frozen", text, count=1)
