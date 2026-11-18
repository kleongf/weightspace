import pytest

from weightspace.target import split_answer
from weightspace.tasks import TASKS, Completion, Example, json_examples


def done(answer: str) -> Completion:
    return Completion(text=answer, answer=answer, finished=True, tokens=10)


def cut_off(text: str) -> Completion:
    return Completion(text=text, answer=None, finished=False, tokens=2048)


GSM = TASKS["gsm8k-answer-line"]
JSON = TASKS["json-extraction"]
SHORT = TASKS["short-answer"]
GSM_EXAMPLE = Example("g", "q", 1200.0)
RECORD = {"name": "Aiko Tanaka", "age": 34, "city": "Osaka", "email": "aiko.tanaka@example.org"}
JSON_EXAMPLE = Example("j", "q", RECORD)
SHORT_EXAMPLE = Example("s", "q", ["George Washington", "Washington"])


@pytest.mark.parametrize(
    ("task", "example", "completion", "strict", "lenient"),
    [
        (GSM, GSM_EXAMPLE, done("So 1,200 total.\nAnswer: 1,200"), 1.0, 1.0),
        (GSM, GSM_EXAMPLE, done("The total is 1200."), 0.0, 1.0),
        (GSM, GSM_EXAMPLE, done("Answer: 1100"), 0.0, 0.0),
        (GSM, GSM_EXAMPLE, cut_off("so 600 + 600 = 1200, wait, so 600 + 600 = 1200"), 0.0, 1.0),
        (JSON, JSON_EXAMPLE, done('{"name": "Aiko Tanaka", "age": 34, "city": "Osaka", "email": "aiko.tanaka@example.org"}'), 1.0, 1.0),
        (JSON, JSON_EXAMPLE, done('```json\n{"name": "Aiko Tanaka", "age": 34, "city": "Osaka", "email": "aiko.tanaka@example.org"}\n```'), 0.0, 1.0),
        (JSON, JSON_EXAMPLE, done('{"name": "Aiko Tanaka", "age": "34", "city": "Kyoto", "email": "aiko.tanaka@example.org"}'), 0.0, 0.75),
        (SHORT, SHORT_EXAMPLE, done("George Washington."), 1.0, 1.0),
        (SHORT, SHORT_EXAMPLE, done("The first president of the United States was George Washington."), 0.0, 1.0),
        (SHORT, SHORT_EXAMPLE, done("Washingtonia"), 0.0, 0.0),
    ],
    ids=["gsm-right", "gsm-no-line", "gsm-wrong", "gsm-looped", "json-exact", "json-fenced", "json-partial", "short-right", "short-wordy", "short-wrong"],
)
def test_strict_and_lenient_scores(task, example, completion, strict, lenient):
    assert task.strict(example, completion) == strict
    assert task.lenient(example, completion) == lenient


def test_reasoning_is_stripped_from_the_answer():
    assert split_answer("let me think</think>\n\nAnswer: 4", thinking=True) == "Answer: 4"


def test_unclosed_reasoning_has_no_answer():
    assert split_answer("let me think and think", thinking=True) is None


def test_json_examples_are_reproducible_and_answerable():
    first, again = json_examples("practice", 20), json_examples("practice", 20)
    assert first == again
    for example in first:
        assert all(str(value) in example.prompt for value in example.reference.values())


@pytest.mark.parametrize("name", sorted(TASKS))
def test_splits_never_share_a_prompt(name):
    task = TASKS[name]
    survey, practice, held_out = (
        {e.prompt for e in task.examples(split, n)} for split, n in [("survey", 40), ("practice", 100), ("held-out", 300)]
    )
    assert not survey & practice and not survey & held_out and not practice & held_out


def test_survey_cannot_reach_into_practice_examples():
    with pytest.raises(ValueError):
        TASKS["gsm8k-answer-line"].examples("survey", 41)
