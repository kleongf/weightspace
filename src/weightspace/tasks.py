import json
import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    reference: Any


@dataclass(frozen=True)
class Completion:
    text: str
    answer: str | None
    finished: bool
    tokens: int


@dataclass(frozen=True)
class Task:
    name: str
    survey_examples: Callable[[int], list[Example]]
    strict: Callable[[Example, Completion], float]
    lenient: Callable[[Example, Completion], float]


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").rstrip("."))
    except ValueError:
        return None


# GSM8K with a required final line

ANSWER_LINE = re.compile(r"^Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def gsm8k_examples(n: int) -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(n))
    return [
        Example(
            id=f"gsm8k-train-{i}",
            prompt=f"{row['question']}\n\nReason step by step, then end with a final line of the form 'Answer: <number>'.",
            reference=as_number(row["answer"].split("####")[-1].strip()),
        )
        for i, row in enumerate(rows)
    ]


def gsm8k_strict(example: Example, completion: Completion) -> float:
    if not completion.finished or completion.answer is None:
        return 0.0
    lines = [line.strip() for line in completion.answer.strip().splitlines() if line.strip()]
    match = ANSWER_LINE.match(lines[-1]) if lines else None
    return float(bool(match) and as_number(match.group(1)) == example.reference)


def gsm8k_lenient(example: Example, completion: Completion) -> float:
    numbers = NUMBER.findall(completion.answer or completion.text)
    return float(bool(numbers) and as_number(numbers[-1]) == example.reference)


# Synthetic JSON extraction

FIRST = ["Maria", "James", "Aiko", "Omar", "Lena", "Diego", "Priya", "Tomas", "Zara", "Felix", "Nadia", "Kofi"]
LAST = ["Lopez", "Carter", "Tanaka", "Haddad", "Novak", "Silva", "Raman", "Berg", "Okafor", "Moreau", "Ivanova", "Mensah"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Krakow", "Toronto", "Valparaiso", "Tallinn", "Hanoi", "Cork"]
TEMPLATES = [
    "{name}, {age}, moved to {city} last spring. You can reach them at {email}.",
    "Contact: {email}. {name} is a {age}-year-old engineer based in {city}.",
    "Our new volunteer from {city} is {name} (age {age}); send questions to {email}.",
    "{name} lives in {city}. Aged {age}, they prefer email: {email}.",
]
JSON_KEYS = ("name", "age", "city", "email")


def json_examples(n: int, seed: str = "survey") -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        record = {
            "name": f"{first} {last}",
            "age": rng.randint(18, 80),
            "city": rng.choice(CITIES),
            "email": f"{first}.{last}@example.{rng.choice(['com', 'org', 'net'])}".lower(),
        }
        text = rng.choice(TEMPLATES).format(**record)
        examples.append(
            Example(
                id=f"json-{seed}-{i}",
                prompt=(
                    f"Text: {text}\n\nExtract a JSON object with exactly these keys: "
                    "name (string), age (integer), city (string), email (string). "
                    "Output only the JSON object, with no other text."
                ),
                reference=record,
            )
        )
    return examples


def json_strict(example: Example, completion: Completion) -> float:
    if not completion.finished or completion.answer is None:
        return 0.0
    try:
        parsed = json.loads(completion.answer.strip())
    except json.JSONDecodeError:
        return 0.0
    return float(parsed == example.reference)


def json_lenient(example: Example, completion: Completion) -> float:
    text = (completion.answer or completion.text).lower()
    found = [str(example.reference[key]).lower() in text for key in JSON_KEYS]
    return sum(found) / len(found)


# Short answers with a word limit

WORD_LIMIT = 5


def normalize(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def contains_alias(aliases: list[str], text: str) -> bool:
    haystack = f" {normalize(text)} "
    return any(f" {normalize(alias)} " in haystack for alias in aliases if normalize(alias))


def short_examples(n: int) -> list[Example]:
    from datasets import load_dataset

    rows = load_dataset("google-research-datasets/nq_open", split="train").shuffle(seed=0).select(range(n))
    return [
        Example(
            id=f"nq-train-{i}",
            prompt=f"Question: {row['question']}?\n\nAnswer in at most {WORD_LIMIT} words.",
            reference=list(row["answer"]),
        )
        for i, row in enumerate(rows)
    ]


def short_strict(example: Example, completion: Completion) -> float:
    if not completion.finished or completion.answer is None:
        return 0.0
    within_limit = len(completion.answer.split()) <= WORD_LIMIT
    return float(within_limit and contains_alias(example.reference, completion.answer))


def short_lenient(example: Example, completion: Completion) -> float:
    return float(contains_alias(example.reference, completion.answer or completion.text))


TASKS = {
    task.name: task
    for task in [
        Task("gsm8k-answer-line", gsm8k_examples, gsm8k_strict, gsm8k_lenient),
        Task("json-extraction", json_examples, json_strict, json_lenient),
        Task("short-answer", short_examples, short_strict, short_lenient),
    ]
}
